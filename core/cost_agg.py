import torch
import torch.nn as nn
import torch.nn.init as init
from libs.GANet.modules.GANet import DisparityRegression
from libs.GANet.modules.GANet import MyNormalize
from libs.GANet.modules.GANet import GetWeights, GetFilters
from libs.GANet.modules.GANet import SGA
from libs.GANet.modules.GANet import NLFIter
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class DomainNorm2(nn.Module):
    def __init__(self, channel, l2=True):
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=False)
        self.l2 = l2
        self.weight = nn.Parameter(torch.ones(1,channel,1,1))
        self.bias = nn.Parameter(torch.zeros(1,channel,1,1))
        self.weight.requires_grad = True
        self.bias.requires_grad = True
    def forward(self, x):
        x = self.normalize(x)
        if self.l2:
            x = F.normalize(x, p=2, dim=1)
        return x * self.weight + self.bias

class DomainNorm(nn.Module):
    def __init__(self, channel, l2=True):
        """ instance norm with option to do l2-normalization over the channels dimension as a first step
            the instance norm has learnable affine parameters for each channel

        Args:
            channel (int): number of channels in the module input tensor
            l2 (bool, optional): whether to apply l2-norm over the channels vector (for each pixel in each batch independently). Defaults to True.
        """
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=True)
        self.l2 = l2
    def forward(self, x):
        if self.l2:
            # divide channel vectors x[b,:,h,w] by their l2-norm
            # now: l2norm(x[b,:,h,w]) = 1
            x = F.normalize(x, p=2, dim=1)
        # instance normalization: E[x] = mean(x[b,c,:,:]), Var[x] = var(x[b,c,:,:])
        # after normalization: mean(x[b,c,:,:]) = 0, var[x[b,c,:,:]] = 1
        x = self.normalize(x)
        return x


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, l2=True, relu=True, **kwargs):
        """ 2d or 3d convolution including optional batch normalization and optional relu activation

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            deconv (bool, optional): use a ConvTranspose2/3d layer instead of Conv2/3d. Defaults to False.
            is_3d (bool, optional): use 3d convolutions instead of 2d ones. Defaults to False.
            bn (bool, optional): use the batch norm. Defaults to True.
            l2 (bool, optional): use the l2 norm for the domain norm. Defaults to True.
            relu (bool, optional): whether to use relu actication. Defaults to True.
        """
        super(BasicConv, self).__init__()
#        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        self.l2 = l2
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = DomainNorm(channel=out_channels, l2=self.l2)
#            self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True, kernel=None):
        super(Conv2x, self).__init__()
        self.concat = concat
        if kernel is not None:
            self.kernel = kernel
        #elif deconv and is_3d: 
        #    kernel = (4, 4, 4)
        
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        
        # used in deconvolution with kernel size 4 and stride 2: doubles height/width
        # stride 2 -> no overlap of kernel additions to output
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)


        if self.concat: 
            self.conv2 = BasicConv(out_channels*2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        #print(x.shape, rem.shape)
        assert(x.size() == rem.size()),[x.size(), rem.size()]
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x

class SGABlock(nn.Module):
    def __init__(self, channels=32, refine=False):
        """ create a semi global aggregation block

        Args:
            channels (int, optional): Number of channels of the . Defaults to 32.
            refine (bool, optional): Whether to add convolution for refinement after sga. Defaults to False.
        """
        super(SGABlock, self).__init__()
        self.refine = refine
        
        # whether to add a convolution after sga
        if self.refine:
            self.bn_relu = nn.Sequential(nn.BatchNorm3d(channels),
                                         nn.ReLU(inplace=True))
            self.conv_refine = BasicConv(channels, channels, is_3d=True, kernel_size=3, padding=1, relu=False)
#            self.conv_refine1 = BasicConv(8, 8, is_3d=True, kernel_size=1, padding=1)
        else:
            self.bn = nn.BatchNorm3d(channels)
        
        # Semi Global Aggregation from GANet
        self.SGA=SGA()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, g):
        """ aggregate a 3d correlation volume (like in stereo matching task)

        Args:
            x (torch.Tensor): 3d correlation volume of shape (batch, wd or ht, ht, wd)
            g (dict): guidance (aggregation weights for each pixel) of shape (batch, 4*5, ht, wd)

        Returns:
            torch.Tensor: _description_
        """
        rem = x

        # split guidance for each direction
        k1, k2, k3, k4 = torch.split(g, (5, 5, 5, 5), 1)
        
        # make the guidance for each direction sum to one
        k1 = F.normalize(k1, p=1, dim=1)
        k2 = F.normalize(k2, p=1, dim=1)
        k3 = F.normalize(k3, p=1, dim=1)
        k4 = F.normalize(k4, p=1, dim=1)

        # apply semi-global aggregation
        x = self.SGA(x, k1, k2, k3, k4)
        
        # apply norm, relu and refinement
        # OR: only norm
        if self.refine:
            x = self.bn_relu(x)
            x = self.conv_refine(x)
        else:
            x = self.bn(x)
        
        assert(x.size() == rem.size())
        
        # add correlation volume before aggregation to correlation volume after aggregation
        x += rem
        
        # apply relu to the sum of aggregated and unaggregated correlation volume
        return self.relu(x)    

def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False
 
class ShiftRegression(nn.Module):
    def __init__(self, max_shift=192):
        super(ShiftRegression, self).__init__()
        self.max_shift = max_shift
#        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1,self.maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x, max_shift=None):
        """ motion regression with weights in x

        Args:
            x (torch.Tensor): softmax of indexed correlation volume of shape (batch, 2*(max_shift//4)+1, 2*ht, 2*wd)
            max_shift (torch.Tensor, optional): absolute maximum of u value. Defaults to None.

        Returns:
            torch.Tensor: regressed flow of shape (batch, 1, 2*ht, 2*wd)
        """
        
        # max_shift of this function receives max_shift//4 of caller ShiftEstimate2
        if max_shift is not None:
            self.max_shift = max_shift
        
        assert(x.is_contiguous() == True)
        
        with torch.cuda.device_of(x):
            
            # create displacements of shape (1, 2*(max_shift//4)+1, 1, 1) using numpy
            shift = Variable(torch.Tensor(np.reshape(np.array(range(-self.max_shift, self.max_shift+1)),[1,self.max_shift*2+1,1,1])).cuda(), requires_grad=False)
            
            # repeat displacements for each batch element and every pixel
            # shape: (batch, 2*(max_shift//4)+1, 2*ht, 2*wd)
            shift = shift.repeat(x.size()[0],1,x.size()[2],x.size()[3])
            
            # multiply tensors of same shape 
            # sum over u value dimension -> regressed u value for the pixel is the result
            # shape: (batch, 1, 2*ht, 2*wd)
            out = torch.sum(x*shift,dim=1,keepdim=True)
        
        return out

class ShiftEstimate(nn.Module):

    def __init__(self, max_shift=192, InChannel=24):
        super(ShiftEstimate, self).__init__()
        self.max_shift = int(max_shift/2)
        self.softmax = nn.Softmax(dim=1)
        self.regression = ShiftRegression(max_shift=self.max_shift+1)
        self.conv3d_2d = nn.Conv3d(InChannel, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=True)
        #self.upsample_cost = FilterUpsample()
    
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/3, W/3, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 3, 3, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(3 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, 3*H, 3*W)

    def forward(self, x):
        #N, _,  _, H, W = g.size()
        #assert (x.size(3)==H and x.size(4)==W)
        #x = self.upsample_cost(x, g)
        #print(x.size(), g.size())
        x = F.interpolate(self.conv3d_2d(x), [self.max_shift*2+1, x.size()[3]*4, x.size()[4]*4], mode='trilinear', align_corners=True)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)
        x = self.regression(x)
        x = F.interpolate(x, [x.size()[2]*2, x.size()[3]*2], mode='bilinear', align_corners=True)
        return x * 2

class ShiftEstimate2(nn.Module):

    def __init__(self, max_shift=100,  InChannel=24):
        """_summary_

        Args:
            max_shift (int, optional): maximum displacement. Defaults to 100.
            InChannel (int, optional): number of channels in the indexed correlation volume. Defaults to 24.
        """
        super(ShiftEstimate2, self).__init__()
        self.max_shift = int(max_shift//4)
        self.softmax = nn.Softmax(dim=1)
        self.regression = ShiftRegression()
        self.conv3d_2d = nn.Conv3d(InChannel, 1, kernel_size=3, stride=1, padding=1, bias=True)
        #self.upsample_cost = FilterUpsample()

    def forward(self, x, max_shift=None):
        """ Apply motion regression, given the indexed correlation volume to calculate the inital flow

        Args:
            x (torch.Tensor): indexed 3d corr volume, shape: (batch, inner_dim, 2*(max_shift//8)+1, ht, wd)
            max_shift (int, optional): maximum displacement. Defaults to None.

        Returns:
            torch.Tensor: full resolution inital flow of shape (batch, 1 HT, WD)
        """
        
        # divide max shift by 4
        if max_shift is not None:
            assert ((max_shift//8 * 2 + 1) == x.shape[2]),[x.shape, max_shift, max_shift//8*2+1]
        #assert(x.size() == rem.size()),[x.size(), rem.size()]
            self.max_shift = max_shift // 4 
        
        # conv3d_2d: reduce channels size to one
        # shape after conv: (batch, 1, 2*(max_shift//8)+1, ht, wd)
        # interpolation doubles size of last three dimensions displacement, height, width
        # shape after interpolate: (batch, 1, 2*(max_shift//4)+1, 2*ht, 2*wd)
        x = F.interpolate(self.conv3d_2d(x), [self.max_shift*2+1, x.size()[3]*2, x.size()[4]*2], mode='trilinear', align_corners=True)
#        x = self.conv3d_2d(x)
        
        # shape after squeeze: (batch, 2*(max_shift//4)+1, 2*ht, 2*wd)
        x = torch.squeeze(x, 1)
       
        # softmax over the displacement/correlation values
        x = self.softmax(x)
        
        # regressed flow
        # shape: (batch, 1, 2*ht, 2*wd)
        x = self.regression(x, self.max_shift)
        
        # interpolate to full-size flow:
        # shape: (batch, 1, 8*ht, 8*wd) = (batch, 1 HT, WD)
        x = F.interpolate(x, [x.size()[2]*4, x.size()[3]*4], mode='bilinear', align_corners=True)

        # scale flow to account for increase in spatial dimension size
        return x * 4


class CostAggregation(nn.Module):
    
    def __init__(self, max_shift=400, in_channel=8):
        super(CostAggregation, self).__init__()
        self.max_shift = max_shift
        self.in_channel = in_channel #t(self.max_shift / 6) * 2 + 1
        self.inner_channel = 8
        self.conv0 = BasicConv(self.in_channel, self.inner_channel, is_3d=True, kernel_size=3, padding=1, relu=True)

        # each of these layers reduces the resolution by half, adding channels
        self.conv1a = BasicConv(self.inner_channel, self.inner_channel*2, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(self.inner_channel*2, self.inner_channel*4, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(self.inner_channel*4, self.inner_channel*6, is_3d=True, kernel_size=3, stride=2, padding=1)

        # each of these layers doubles the spatial resolution, removing channels
        self.deconv1a = Conv2x(self.inner_channel*2, self.inner_channel, deconv=True, is_3d=True, relu=True)
        self.deconv2a = Conv2x(self.inner_channel*4, self.inner_channel*2, deconv=True, is_3d=True)
        self.deconv3a = Conv2x(self.inner_channel*6, self.inner_channel*4, deconv=True, is_3d=True)

        # each of these layers reduces the resolution by half, adding channels
        self.conv1b = BasicConv(self.inner_channel, self.inner_channel*2, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.conv2b = BasicConv(self.inner_channel*2, self.inner_channel*4, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.conv3b = BasicConv(self.inner_channel*4, self.inner_channel*6, is_3d=True, kernel_size=3, stride=2, padding=1)

        # each of these layers doubles the spatial resolution, removing channels
        self.deconv1b = Conv2x(self.inner_channel*2, self.inner_channel, deconv=True, is_3d=True, relu=True, kernel=(3,4,4))
        self.deconv2b = Conv2x(self.inner_channel*4, self.inner_channel*2, deconv=True, is_3d=True, kernel=(3,4,4))
        self.deconv3b = Conv2x(self.inner_channel*6, self.inner_channel*4, deconv=True, is_3d=True, kernel=(3,4,4))
        
        # Motion Regression
        self.shift0= ShiftEstimate2(self.max_shift, self.inner_channel)
        self.shift1= ShiftEstimate2(self.max_shift, self.inner_channel)
        self.shift2= ShiftEstimate2(self.max_shift, self.inner_channel)
        
        # Semi Global Aggregation Blocks for weights sg1, sg2, sg3, sg11, sg12
        self.sga1 = SGABlock(channels=self.inner_channel, refine=True) 
        self.sga2 = SGABlock(channels=self.inner_channel, refine=True) 
        self.sga3 = SGABlock(channels=self.inner_channel, refine=True) 
        self.sga11 = SGABlock(channels=self.inner_channel*2, refine=True) 
        self.sga12 = SGABlock(channels=self.inner_channel*2, refine=True) 
        
        # Reduces number of channels to 1 for final correlation volume output
        self.corr_output = BasicConv(self.inner_channel, 1, is_3d=True, kernel_size=3, padding=1, relu=False)
        
        # responsible of 1d correlation volume indexing, required for motion regression
        self.corr2cost = Corr2Cost()
    
    def forward(self, x, g, max_shift=400, is_ux=True):
        """ aggregates the C_u and C_v volumes with shape (batch, 2*levels, wd OR ht, ht, wd)


        Args:
            x (torch.Tensor): C_u or C_v cost volume of shape (batch, 2*levels, wd OR ht, ht, wd)
            g (dict): contains sg1, sg2, sg3, sg11, sg12 tensors for guiding the aggregation based on image1 and its features
            max_shift (int, optional): largest possible shift (u OR v displacement) used in motion regression. Defaults to 400.
            is_ux (bool, optional): True if C_u is aggregated, otherwise False. Defaults to True.

        Returns:
            if self.training:
                Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor] : shift0, shift1, shift2, corr
                shift0  :   motion regression from first preliminary 3D correlation volume
                shift1  :   motion regression from second preliminary 3D correlation volume
                shift2  :   motion regression from final aggregated 3D correlation volume
                corr    :   final aggregated 3D correlation volume
            else:
                Tuple[torch.Tensor,torch.Tensor] : shift2, corr
        """
        
        # receives (batch, 2*levels, wd OR ht, ht, wd)
        # shape: (batch, self.inner_channel, wd OR ht, ht, wd)
        x = self.conv0(x)
    
        # first aggregation step
        # shape: (batch, self.inner_channel, wd OR ht, ht, wd)
        x = self.sga1(x, g['sg1'])
        rem0 = x

        # shift0 is calculated from the least-aggregated cost volume by motion regression
        # shift0 is the full-resolution, motion regressed flow of one direction (u or v)
        # this is used to calculate an additional loss
        if self.training:
            # index correlation volume
            # shape: (batch, inner_dim, 2*(max_shift//8)+1, ht, wd)
            cost = self.corr2cost(x, max_shift//8, is_ux)
            shift0 = self.shift0(cost, max_shift)

        # hourglass network with skip-connections and semi-global aggregation:
        #   l0:         (batch,     1*self.inner_channel,   wd OR ht,   ht,     wd)
        #   d1:         (batch,     2*self.inner_channel,   wd OR ht,   ht//2,  wd//2)
        #   d2:         (batch,     4*self.inner_channel,   wd OR ht,   ht//4,  wd//4)
        #   d3:         (batch,     6*self.inner_channel,   wd OR ht,   ht//8,  wd//8)
        #   u1+d2:      (batch,     4*self.inner_channel,   wd OR ht,   ht//4,  wd//4)
        #   agg(u2+d1): (batch,     2*self,inner_channel,   wd OR ht,   ht//2,  wd//2)
        #   agg(u3+l0): (batch,     1*self,inner_channel,   wd OR ht,   ht,     wd)

        # shape: (batch, 2*self.inner_channel, wd OR ht, ht//2, wd//2)
        x = self.conv1a(x)
        x = self.sga11(x, g['sg11'])
        rem1 = x
        
        # shape: (batch, 4*self.inner_channel, wd OR ht, ht//4, wd//4)
        x = self.conv2a(x)
        rem2 = x
        
        # shape: (batch, 6*self.inner_channel, wd OR ht, ht//8, wd//8)
        x = self.conv3a(x)
        rem3 = x

        # shape: (batch, 4*self.inner_channel, wd OR ht, ht//4, wd//4)
        x = self.deconv3a(x, rem2)
        rem2 = x

        # shape: (batch, 2*self.inner_channel, wd OR ht, ht//2, wd//2)
        x = self.deconv2a(x, rem1)
        x = self.sga12(x, g['sg12'])
        rem1 = x
        
        # shape: (batch, self.inner_channel, wd OR ht, ht, wd)
        x = self.deconv1a(x, rem0)
        x = self.sga2(x, g['sg2'])
        rem0 = x
        
        # index correlation volume
        # shape: (batch, inner_dim, 2*(max_shift//8)+1, ht, wd)
        cost = self.corr2cost(x, max_shift//8, is_ux)
        
        # motion regression to mix all correlations
        # shift1 is the full-resolution, motion regressed flow of one direction (u or v)
        # this is used to calculate an additional loss
        if self.training:
            shift1 = self.shift1(cost, max_shift)
        
        # final correlation volume for one direction (C^A_u or C^A_v)
        # convolution reduces the channel size from inner_channel to 1
        # shape: (batch, 1, wd OR ht, ht, wd)
        corr = self.corr_output(x) 
        
        # another hourglass module
        # interestingly, this processes the indexed 3d correlation volume
        # shape: (batch, inner_dim, 2*(max_shift//8)+1, ht, wd)
        rem0 = cost
        x = self.conv1b(cost)
        rem1 = x
        x = self.conv2b(x)
        rem2 = x
        x = self.conv3b(x)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)
        x = self.sga3(x, g['sg3'])
        
        # the hourglass-refined indexed 3d correlation volume is directly used for motion regression
        # shift2 is the resulting, final inital motion-regressed cost volume
        shift2 = self.shift2(x, max_shift)
        
        if self.training:
            #      u0      u1      flow_u  (C^A_u OR C^A_v)
            return shift0, shift1, shift2, corr
        else:
            #      flow_u  (C^A_u OR C^A_v)
            return shift2, corr

class Corr2Cost(nn.Module):
    
    def __init__(self):
        super(Corr2Cost, self).__init__()
    
    def coords_grid(self, batch, ht, wd, device):
        """ create coordinate grid
            exactly the same as core.utils.coords_grid, but with custom device

        Args:
            batch (int): batch size
            ht (int): image height
            wd (int): image width
            device (torch.Device): device to run computation

        Returns:
            torch.Tensor: coordinate grid of shape (batch, 2, ht, wd)
        """
        coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)
    
    def bilinear_sampler(self, img, coords, mode='bilinear', mask=False):
        """ Wrapper for grid_sample, uses pixel coordinates
        
        Args:
            img (torch.Tensor): correlation volume of shape (batch*h1*w1, dim, h2, w2)
            coords (torch.Tensor): coordinates to sample for each pixel of shape (batch*h1*w1, 1, 2*r+1, 2)
            mode (str, optional): Not accessed by program currently - could be passed to grid_sample. Defaults to 'bilinear'.
            mask (bool, optional): Whether to return the mask being 1.0 for pixels outside of grid,
                                    same shape as return value. Defaults to False.
        Returns:
            torch.Tensor: sampled correlation values for the specified coordinates, shape (batch*h1*w1, dim, 1, 2*r+1)
                            correlation values sampled around the current estimated image2 location for each image1 pixel
        """

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

    def forward(self, corr, maxdisp=50, is_ux=True):
        """ index 3d correlation volume with all possible displacements for u OR v (up to maxdisp)

        Args:
            corr (torch.Tensor): 3d correlation volume of shape (batch, inner_dim, wd OR ht, ht, wd)
            maxdisp (int, optional): largest displacement considered for indexing. Defaults to 50.
            is_ux (bool, optional): whether the corr stems from C_u or C_v. Defaults to True.

        Returns:
            torch.Tensor: indexed 3d correlation volumes
                correlations for all displacements for each u-value at each pixel
                shape: (batch, inner_dim, 2*maxdisp+1, ht, wd)
        """
        batch, dim, d, h, w = corr.shape
        
        # permute: (batch, inner_dim, wd OR ht, ht, wd)
        # reshape: (batch*ht*wd, inner_dim, 1, wd OR ht)
        corr = corr.permute(0, 3, 4, 1, 2).reshape(batch*h*w, dim, 1, d)
        
        # create coordinates to index 3d correlation volume C_u OR C_v
        with torch.no_grad():
            coords = self.coords_grid(batch, h, w, corr.device)
            
            # select:
            #   u coords: (batch, 1, ht, wd)
            #   v coords: (batch, 1, ht, wd)
            if is_ux:
                coords = coords[:, :1, :, :]
            else:
                coords = coords[:, 1:, :, :]

            # create u/v displacement tensor with all integer possibilities:
            #   [-maxdisp, -maxdisp+1, ..., -1, 0, 1, ..., maxdisp-1, maxdisp]
            dx = torch.linspace(-maxdisp, maxdisp, maxdisp*2+1)
            
            # reshape: (1, 1, 2*maxdisp+1, 1)
            dx = dx.view(1, 1, 2*maxdisp+1, 1).to(corr.device)
            
            # broadcast displacements over u coords
            # creating tensor with displacements for each u coordinate
            # dx: (1, 1, 2*maxdisp+1, 1) + coords: (batch*h*w, 1, 1, 1)
            #   -> x0: (batch*h*w, 1, 2*maxdisp+1, 1)
            x0 = dx + coords.reshape(batch*h*w, 1, 1, 1)
            
            # y has to be zero since the other (non v OR u) dim has size 1
            # shape: (batch*h*w, 1, 2*maxdisp+1, 1)
            y0 = torch.zeros_like(x0)
           
           # if is_ux:
            
            # concatenate 0-valued y0 to x0 coord displacements
            # shape: (batch*h*w, 1, 2*maxdisp+1, 2)
            coords_lvl = torch.cat([x0,y0], dim=-1)
           
           # else:
           #     coords_lvl = torch.cat([y0, x0], dim=-1)
        
        # sample from 1d correlation volume with u displacements
        # shape: (batch*ht*wd, inner_dim, 1, 2*maxdisp+1)
        corr = self.bilinear_sampler(corr, coords_lvl)
        
        #print(corr.shape)
        
        # shape: (batch*ht*wd, inner_dim, 1, 2*maxdisp+1)
        #   -> (batch, ht, wd, inner_dim, 2*maxdisp+1)
        corr = corr.view(batch, h, w, dim, maxdisp*2+1)

        # permute: (batch, inner_dim, 2*maxdisp+1, ht, wd)
        corr = corr.permute(0, 3, 4, 1, 2).contiguous().float()
        
        return corr

