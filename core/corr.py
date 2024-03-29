import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.utils import bilinear_sampler, coords_grid
from libs.MemorySaver.functions import ComputeMaxAvgFunction, ComputeMaxArgmaxAvgFunction
from libs.MemorySaver.functions import ComputeSelfCompressionFunction
from core.nlf import NLF

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

class CorrBlock:
    """ Module for correlation volume creation, aggregation, separation, 
        pyramid pooling and lookup
    """
    def __init__(self, fmap1, fmap2, guid, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # only create NLF if guidance is provided
        if guid is not None:
            self.nlf = NLF()
        # all pairs correlation
        corr = self.corr_compute(fmap1, fmap2, guid, reverse=True)

        batch, h1, w1, h2, w2 = corr.shape
        self.shape = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, h2, w2)
        
        # create 4D correlation volume pyramid
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)
    
    def separate(self, attention_weights):
        """ create two separate correlation volumes 
            in this version, they only consist of the max and avg
            -> no use of self-adaptive compression, because only attention in vector
            -> no additional fields aggregated through attention

            they are interpolated on every level to have the original ht/wd dimension

            shape:
                sep_u: (batch, 2*levels, wd, ht, wd)
                sep_v: (batch, 2*levels, ht, ht, wd)

            confusing naming scheme between paper and implementation:
            implementation      paper
            sep_u           ->  C_v
            sep_v           ->  C_u

        Args:
            attention_weights (Tuple[torch.nn.Conv3d,torch.nn.Conv3d]):
                3d convolution for computing attention weights for corr1 and corr2
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 3d correlation volumes corr1 (sep_u) and corr2 (sep_v)
        """
        batch, h1, w1, h2, w2 = self.shape

        sep_u_lvls = []
        sep_v_lvls = []

        # for each level of 4d correlation volume pyramid
        for i in range(self.num_levels):
            # correlation volume at level i
            # shape: (batch*ht*wd, 1, ht/2**i, wd/2**i)
            corr = self.corr_pyramid[i]
            
            # max values of correlation volume
            # shape: (batch*ht*wd, 1, 1, wd/2**i)
            m1, _ = corr.max(dim=2, keepdim=True)
            
            # avg values of correlation volume
            # shape: (batch*ht*wd, 1, 1, wd/2**i)
            m2 = corr.mean(dim=2, keepdim=True)
            
            # sep_u is C_v, because of its last dimension size depends on wd
            # this is the basis of the attention vector from the paper
            # sep only contains max and avg, no additional attention-based fields like in paper
            # (batch*ht*wd, 1, 1, wd/2**i) (batch*ht*wd, 1, 1, wd/2**i) -> (batch*ht*wd, 1, 2, wd/2**i)
            sep_u = torch.cat((m1, m2), dim=2)
            
            # reshape: (batch*ht*wd, 1, 2, wd/2**i)     -> (batch, ht, wd, 2, wd/2**i)
            # permute: (batch, ht, wd, 2, wd/2**i)   -> (batch, 2, wd/2**i, ht, wd)
            sep_u = sep_u.reshape(batch, h1, w1, sep_u.shape[2], sep_u.shape[3]).permute(0, 3, 4, 1, 2)


            # exactly the same for the width dimension
            # (batch*ht*wd, 1, ht/2**i, 1)
            m1, _ = corr.max(dim=3, keepdim=True)
            # (batch*ht*wd, 1, ht/2**i, 1)
            m2 = corr.mean(dim=3, keepdim=True)
            # (batch*ht*wd, 1, ht/2**i, 2)
            sep_v = torch.cat((m1, m2), dim=3)
            # (batch, 2, ht/2**i, ht, wd)
            sep_v = sep_v.reshape(batch, h1, w1, sep_v.shape[2], sep_v.shape[3]).permute(0, 4, 3, 1, 2)


            if attention_weights is not None:

                attention1, attention2 = attention_weights

                # shape: (batch*ht*wd, 1, ht/2**i, wd/2**i) -> (batch, ht, wd, 1, ht/2**i, wd/2**i)
                shaped_corr = corr.view((batch, h1, w1, corr.shape[1], corr.shape[2], corr.shape[3]))
                
                # shape: (batch, ht, wd, 1, ht/2**i, wd/2**i) -> (batch, 1, ht/2**i, wd/2**i, ht, wd)
                shaped_corr = shaped_corr.permute((0,3,4,5,1,2))

                # shape: (batch, 2, wd/2**i, ht, wd) -> (batch, corr_channels-2, wd/2**i, ht, wd)
                a_u = attention1(sep_u)

                # shape: (batch, corr_channels-2, wd/2**i, ht, wd) -> (batch, corr_channels-2, 1, wd/2**i, ht, wd)
                a_u = a_u.unsqueeze(dim=2)

                # apply softmax over v-dimension
                a_u = a_u.softmax(dim=3)
                
                # shape:
                #       (batch, corr_channels-2,    1,          wd/2**i, ht, wd)    attention
                #   *   (batch, 1,                  ht/2**i,    wd/2**i, ht, wd)    4d correlation volume
                #   ->  (batch, corr_channels-2,    ht/2**i,             ht, wd)    
                adaptive_corr_u = torch.einsum('bcuvij,bcuvij->bcuij',a_u, shaped_corr)

                # shape: (batch, 2, ht/2**i, ht, wd) -> (batch, corr_channels-2, ht/2**i, ht, wd)
                a_v = attention2(sep_v)
                # shape: (batch, corr_channels-2, ht/2**i, ht, wd) -> (batch, corr_channels-2, ht/2**i, 1, ht, wd)
                a_v = a_v.unsqueeze(dim=3)
                a_v = a_v.softmax(dim=2)

                # shape:
                #       (batch, corr_channels-2,    ht/2**i,    1,          ht, wd)    attention
                #   *   (batch, 1,                  ht/2**i,    wd/2**i,    ht, wd)    4d correlation volume
                #   ->  (batch, corr_channels-2,                wd/2**i,    ht, wd)
                adaptive_corr_v = torch.einsum('bcuvij,bcuvij->bcvij',a_v, shaped_corr)

                # shape: (batch, corr_channels, ht/2**i, ht, wd)
                sep_v = torch.cat((sep_v, adaptive_corr_u), dim=1)
                # shape: (batch, corr_channels, wd/2**i, ht, wd)
                sep_u = torch.cat((sep_u, adaptive_corr_v), dim=1)

            # maybe they used the largest-level correlation volume to pair the attention with
            # also, upsampling is the only way the concatenation at the end is possible (otherwise, shape[2] would not match)
            # (batch, corr_channels, wd/2**i, ht, wd) -> (batch, corr_channels, wd, ht, wd)
            sep_u = F.interpolate(sep_u, [w2, h1, w1], mode='trilinear', align_corners=True)

            sep_u_lvls.append(sep_u)

            # (batch, corr_channels, ht, ht, wd)
            sep_v = F.interpolate(sep_v, [h2, h1, w1], mode='trilinear', align_corners=True)

            sep_v_lvls.append(sep_v)
        
        # concatenate over all levels
        # list -> (batch, corr_channels*levels, wd, ht, wd)
        sep_u_lvls = torch.cat(sep_u_lvls, dim=1)
        # list -> (batch, corr_channels*levels, ht, ht, wd)
        sep_v_lvls = torch.cat(sep_v_lvls, dim=1)
        
        return sep_u_lvls, sep_v_lvls


    def __call__(self, coords, sep=False, attention_weights=None):
        """ applies the lookup operation to all pyramid levels

        Args:
            coords (torch.Tensor): displacement location in image 2 feature map
            sep (bool, optional): whether to separate the cost volume. Defaults to False.
            attention_weights (Tuple[torch.nn.Conv3d, torch.nn.Conv3d], optional): two attention networks. Defaults to None.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor,torch.Tensor]] : 
                separated 3D cost volumes OR lookup of 4D correlation volume pyramid
                depending on sep=True     OR sep=False
        """
        
        # returns two 3d cost volumes
        # sep_u: (batch, ht, wd, ht), sep_v: (batch, ht, wd, wd)
        # very strange behaviour:
        #   if sep==False:
        #       return lookup of 4d correlation volume
        #   if sep==True:
        #       return two 3d correlation volumes without applying the lookup operation
        #       they need to be lookuped in CorrBlock1D still
        if sep:
            return self.separate(attention_weights)
        
        # this part is the same as RAFT
        r = self.radius

        # permutation: (batch, 2, ht, wd) -> (batch, ht, wd, 2)
        coords = coords.permute(0, 2, 3, 1)

        # get batch, ht, wd sizes
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            # get correlation volume at the current level
            corr = self.corr_pyramid[i]
            # get local grid coordinates relative to pixel center
            # dx = dy = [-r, -r+1, ..., -1, 0, 1, ..., r-1, r]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)

            # meshgrid of local coordinates: reversed since dy first
            # fitting since coords is also reversed with y first
            # this has shape (2*r+1, 2*r+1, 2)
            # implies use of infinity norm ||dx||_inf in contrast to paper with 1-norm ||dx||_1
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            # shape (batch, ht, wd, 2) -> (batch*ht*wd, 1, 1, 2)
            # diveded by 2**pyramid_level accounting for the grid step size at each level
            # due to pooling and reduction in image size
            # delta is not diveded -> larger window at higher pyramid_level
            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i

            # reshape (2*r+1, 2*r+1, 2) -> (1, 2*r+1, 2*r+1, 2)
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)

            # coords (batch*ht*wd, 2*r+1, 2*r+1, 2) = centroid (batch*ht*wd, 1, 1, 2) + delta (1, 2*r+1, 2*r+1, 2)
            # for each pixel in image1, there are (2*r+1)**2 coordinates of locations in image2
            coords_lvl = centroid_lvl + delta_lvl

            # the bilinear sampler maps (2*r+1)**2 correlation values to each pixel
            # by using bilinear interpolation on the current correlation pyramid level (pooled values)
            # shape: (batch*h1*w1, 2*r+1, 2*r+1)
            corr = bilinear_sampler(corr, coords_lvl)

            # separate dimensions
            # (batch*h1*w1, dim, 2*r+1, 2*r+1) -> (batch, h1, w1, dim*(2*r+1)*(2*r+1))
            corr = corr.view(batch, h1, w1, -1)

            out_pyramid.append(corr)

        # the correlation values of all pyramid levels are concatenated
        # resulting shape: (batch, h1, w1, num_levels*dim*(2*r+1)*(2*r+1))
        out = torch.cat(out_pyramid, dim=-1)

        # permutation:
        # (batch, h1, w1, num_levels*dim*(2*r+1)*(2*r+1)) -> (batch, num_levels*dim*(2*r+1)*(2*r+1), h1, w1)
        # also contiguous is used to copy the tensor with new memory layout according to shape
        return out.permute(0, 3, 1, 2).contiguous().float()

    #@staticmethod
    def corr_compute(self, fmap1, fmap2, guid, reverse=True):
        """ compute the 4d correlation volume

        Args:
            fmap1 (torch.Tensor): features of image1 of shape (batch, fdim, ht, wd)
            fmap2 (torch.Tensor): features of image2 of shape (batch, fdim, ht, wd)
            guid (torch.Tensor):    guidance weights for 4d cost volume aggregation
                                    if None then 4d cost volume will not be aggregated
            reverse (bool, optional):   matmul(fmap2, fmap1) if True else matmul(fmap1,fmap2).
                                        Defaults to True.

        Returns:
            torch.Tensor: 4d correlation volume of shape (batch, ht, wd, ht, wd)
        """
        batch, dim, ht, wd = fmap1.shape
        
        # flatten height/width dimension
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        # reverse has matmul(fmap2, fmap1) instead of matmul(fmap1, fmap2)
        if reverse:
            
            # see non-reverse branch
            corr = torch.matmul(fmap2.transpose(1,2), fmap1) / torch.sqrt(torch.tensor(dim).float())
            corr = corr.view(batch, ht, wd, ht, wd)
            if guid is not None:
                corr = self.nlf(corr, guid)
            
            # swapping the fmap1 height/width dimensions back to the front:
            # (batch, ht_fmap2, wd_fmap2, ht_fmap1, wd_fmap1) 
            # -> (batch, ht_fmap1, wd_fmap1, ht_fmap2, wd_fmap2)
            corr = corr.permute(0, 3, 4, 1, 2)

        # non-reverse has matmul(fmap1, fmap2) as in raft
        else:
            # standart 4d correlation volume computation from raft
            # matmul: (batch, ht*wd, feature_dim) (batch, feature_dim, ht*wd) -> (batch, ht*wd, ht*wd)
            corr = torch.matmul(fmap1.transpose(1,2), fmap2) / torch.sqrt(torch.tensor(dim).float())
            
            # reshape to separate height/width dimensions
            corr = corr.view(batch, ht, wd, ht, wd)
            
            # cost volume is only aggregated if the function received guidance parameters
            if guid is not None:
                # nlf seems to somehow combine the 4d correlation volume with guidance 
                # guidance is a combination of fmap1 and img1
                # nlf has an up, down, left and right component
                # shape: (batch, ht, wd, ht, wd)
                corr = self.nlf(corr, guid)

        return corr


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())

class AlternateCorrBlockSepflow:
    """ Class for the alternate backward pass of Separable Flow
        Implements the correlation volume separation without storing the 4D correlation volume 
        Does not implement the 3D lookup operation: This is implemented by CorrBlock1D
    """
    def __init__(self, fmap1, fmap2, guid, num_levels=4, radius=4, support_backward=False):
        """ Initialize the separation module

        Args:
            fmap1 (torch.Tensor):   features of image one of shape (batch, fdim, ht, wd)
            fmap2 (_type_):         features of image one of shape (batch, fdim, ht, wd)
            guid (torch.Tensor):    guidance weights for 4d cost volume aggregation
                                    if None then 4d cost volume will not be aggregated
            num_levels (int, optional): number of levels of the 4D correlation volume pyramid. Defaults to 4.
            radius (int, optional): lookup radius: unused. Defaults to 4.
            support_backward (bool, optional): whether to enable the support for the backward pass. Defaults to False.

        Raises:
            Exception: in case guidance NLF weights are passed, e.g. guid is not None
        """

        if guid is not None:
            # NLF does not work with this, since it creates dependencies between the 4D correlation volume indices
            raise Exception("Cannot use NLF while not storing the 4d correlation volume")

        # whether to support the backward pass
        if support_backward:
            # if True: compute argmax alongside max and avg
            self.max_avg_function = ComputeMaxArgmaxAvgFunction
        else:
            # only compute max and avg
            self.max_avg_function = ComputeMaxAvgFunction

        # number of pyramid levels
        self.num_levels = num_levels

        # lookup radius
        self.radius = radius

        self.fmap1_l0 = fmap1

        # build image2 feature pyramid, first level 0 is fmap2
        self.pyramid = [fmap2]
        # average pooling to reduce displacement dimension sizes
        # which correspond to the spatial dimension of fmap2
        for i in range(self.num_levels-1):
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append(fmap2)
    
    def separate(self, attention_weights):
        """ create two separate correlation volumes 
            in this version, they only consist of the max and avg
            -> no use of self-adaptive compression, because only attention in vector
            -> no additional fields aggregated through attention

            they are interpolated on every level to have the original ht/wd dimension

            shape:
                sep_u: (batch, 2*levels, wd, ht, wd)
                sep_v: (batch, 2*levels, ht, ht, wd)

            confusing naming scheme between paper and implementation:
            implementation      paper
            sep_u           ->  C_v
            sep_v           ->  C_u

        Args:
            attention_weights (Tuple[torch.nn.Conv3d,torch.nn.Conv3d]):
                3d convolution for computing attention weights for corr1 and corr2
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 3d correlation volumes corr1 (sep_u) and corr2 (sep_v)
        """

        batch, fdim, h1, w1 = self.fmap1_l0.shape
        h2, w2 = (h1, w1)

        sep_u_lvls = []
        sep_v_lvls = []

        # for each level of 4d correlation volume pyramid
        for i in range(self.num_levels):
            # img1 features at level 0
            # shape: (batch, ht, wd, fdim)
            fmap1 = self.fmap1_l0.permute(0,2,3,1).contiguous() / torch.sqrt(torch.tensor(fdim).float())
            # img2 features at level i
            # shape: (batch, ht/2**i, wd/2**i, fdim)
            fmap2 = self.pyramid[i].permute(0,2,3,1).contiguous()
            
            # shape: (batch, ht, wd, 2, ht/2**i), (batch, ht, wd, 2, wd/2**i)
            maxavg_u, maxavg_v = self.max_avg_function.apply(fmap1, fmap2)
            
            # shape: (batch, ht, wd, 2, ht/2**i) -> (batch, 2, ht/2**i, ht, wd)
            sep_v = maxavg_u.permute(0,3,4,1,2).contiguous()
            sep_u = maxavg_v.permute(0,3,4,1,2).contiguous()


            if attention_weights is not None:

                attention1, attention2 = attention_weights

                # shape: (batch, 2, wd/2**i, ht, wd) -> (batch, corr_channels-2, wd/2**i, ht, wd)
                a_u = attention1(sep_u)
                a_v = attention2(sep_v)

                # apply softmax over v-dimension
                a_u = a_u.softmax(dim=2)
                a_v = a_v.softmax(dim=2)

                # shape: (batch, corr_channels-2, wd/2**i, ht, wd) -> (batch, ht, wd, corr_channels-2, wd/2**i)
                a_u = a_u.permute(0,3,4,1,2).contiguous()
                a_v = a_v.permute(0,3,4,1,2).contiguous()

                adaptive_corr_u, adaptive_corr_v = ComputeSelfCompressionFunction.apply(fmap1, fmap2, a_u, a_v)
                
                # shape: (batch, ht, wd, corr_channels-2, wd/2**i) -> (batch, corr_channels-2, wd/2**i, ht, wd)
                adaptive_corr_u = adaptive_corr_u.permute(0,3,4,1,2)
                adaptive_corr_v = adaptive_corr_v.permute(0,3,4,1,2)

                # shape: (batch, corr_channels, ht/2**i, ht, wd)
                sep_v = torch.cat((sep_v, adaptive_corr_u), dim=1)
                # shape: (batch, corr_channels, wd/2**i, ht, wd)
                sep_u = torch.cat((sep_u, adaptive_corr_v), dim=1)

            # maybe they used the largest-level correlation volume to pair the attention with
            # also, upsampling is the only way the concatenation at the end is possible (otherwise, shape[2] would not match)
            # (batch, corr_channels, wd/2**i, ht, wd) -> (batch, corr_channels, wd, ht, wd)
            sep_u = F.interpolate(sep_u, [w2, h1, w1], mode='trilinear', align_corners=True)

            sep_u_lvls.append(sep_u)

            # (batch, corr_channels, ht, ht, wd)
            sep_v = F.interpolate(sep_v, [h2, h1, w1], mode='trilinear', align_corners=True)

            sep_v_lvls.append(sep_v)
        
        # concatenate over all levels
        # list -> (batch, corr_channels*levels, wd, ht, wd)
        sep_u_lvls = torch.cat(sep_u_lvls, dim=1)
        # list -> (batch, corr_channels*levels, ht, ht, wd)
        sep_v_lvls = torch.cat(sep_v_lvls, dim=1)
        
        return sep_u_lvls, sep_v_lvls

    def __call__(self, coords, sep=False, attention_weights=None):       
        # returns two 3d cost volumes
        # sep_u: (batch, ht, wd, ht), sep_v: (batch, ht, wd, wd)
        # very strange behaviour:
        #   if sep==False:
        #       return lookup of 4d correlation volume
        #   if sep==True:
        #       return two 3d correlation volumes without lookup applied
        #       they need to be lookuped in CorrBlock1D still
        if sep:
            return self.separate(attention_weights)
        
        raise NotImplementedError("Missing the cpp/cuda part for lookup" + 
            "of the 4d correlation volume without storing it")

        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
        

class CorrBlock1D:
    """ lookup operation for the 3D correlation volumes
    """

    def __init__(self, corr1, corr2, num_levels=4, radius=4):
        """ responsible for lookup of the 3d correlation volumes for u and v

        Args:
            corr1 (torch.Tensor): 3D correlation volume 1
            corr2 (torch.Tensor): 3D correlation volume 2
            num_levels (int, optional): number of pyramid levels. Defaults to 4.
            radius (int, optional): lookup radius. Defaults to 4.
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid1 = []
        self.corr_pyramid2 = []

        corr1 = corr1.permute(0,3,4,1,2)
        corr2 = corr2.permute(0,3,4,1,2)
        batch, h1, w1, dim, w2 = corr1.shape
        batch, h1, w1, dim, h2 = corr2.shape
        assert(corr1.shape[:-1] == corr2.shape[:-1])
        assert(h1 == h2 and w1 == w2)

        corr1 = corr1.reshape(batch*h1*w1, dim, 1, w2)
        corr2 = corr2.reshape(batch*h1*w1, dim, 1, h2)

        # build 3D correlation volume pyramids by downsampling
        # across displacement dimension
        self.corr_pyramid1.append(corr1)
        self.corr_pyramid2.append(corr2)
        for i in range(self.num_levels):
            corr1 = F.avg_pool2d(corr1, [1,2], stride=[1,2])
            self.corr_pyramid1.append(corr1)
            corr2 = F.avg_pool2d(corr2, [1,2], stride=[1,2])
            self.corr_pyramid2.append(corr2)

    def bilinear_sampler(self, img, coords, mode='bilinear', mask=False):
        """ Wrapper for grid_sample, uses pixel coordinates """
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1,1], dim=-1)
        xgrid = 2*xgrid/(W-1) - 1
        assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem

        grid = torch.cat([xgrid, ygrid], dim=-1)
        img = F.grid_sample(img, grid, align_corners=True)

        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask.float()

        return img

    def __call__(self, coords):
        """ perform lookup on both 3D correlation volumes and all pyramid levels

        Args:
            coords (torch.Tensor): current estimated displacement locations in fmap2

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: lookup of both correlation volumes concatenated 
            across pyramid levels
        """

        coords_org = coords.clone()
        coords = coords_org[:, :1, :, :]
        coords = coords.permute(0, 2, 3, 1)
        r = self.radius
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid1[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1)
            coords_lvl = torch.clamp(coords_lvl, -1, 1)

            corr = self.bilinear_sampler(corr, coords_lvl)

            corr = corr.view(batch, h1, w1, -1)

            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out1 = out.permute(0, 3, 1, 2).contiguous().float()

        coords = coords_org[:, 1:, :, :]
        coords = coords.permute(0, 2, 3, 1)
        r = self.radius
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid2[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0, y0], dim=-1)
            corr = self.bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out2 = out.permute(0, 3, 1, 2).contiguous().float()
        return out1, out2
