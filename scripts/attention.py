from typing import Callable
import torch
import torch.nn.functional as F
from datetime import datetime
from datetime import timedelta

class CorrBlock:
    
    def __init__(self, batch, ht, wd, num_levels, radius):
        
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = torch.randn((batch, ht, wd, ht, wd))

        batch, h1, w1, h2, w2 = corr.shape
        self.shape = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

        self.junklist = []


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
            attention (Tuple[torch.nn.Conv3d,torch.nn.Conv3d]):
                convolution for computing attention weights for corr1 and corr2
        
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
                #   ->  (batch, corr_channels-2,    ht/2**i,    wd/2**i, ht, wd)    
                adaptive_corr_u = a_u * shaped_corr

                # shape:    (batch, corr_channels-2,    ht/2**i,    wd/2**i,    ht, wd)
                #   ->      (batch, corr_channels-2,    ht/2**i,    ht, wd)
                adaptive_corr_u = adaptive_corr_u.sum(dim=3)

                print(f"start_test {a_u.shape[1]} {corr.shape[2]}")
                # test if computation is same as paper
                # shape: (batch, corr_channels-2,    ht/2**i,    ht, wd)
                adaptive_corr_u_alt = torch.zeros((batch, a_u.shape[1], corr.shape[2], h1, w1))
                print(adaptive_corr_u_alt.shape)
                print(a_u.shape)
                print(shaped_corr.shape)
                for bn in range(batch):
                    for channel in range(a_u.shape[1]):
                        for i_idx in range(h1):
                            for j_idx in range(w1):
                                for u in range(corr.shape[2]):
                                    
                                    adaptive_corr_u_alt[bn,channel,u,i_idx,j_idx] += a_u[bn,channel,0,:,i_idx,j_idx].dot(shaped_corr[bn,0,u,:,i_idx,j_idx])

                # results are about the same with numerical error margin of 1e-6
                print(((adaptive_corr_u - adaptive_corr_u_alt).abs() > 1e-6).sum()/adaptive_corr_u_alt.flatten().shape[0])
                self.junklist.append(adaptive_corr_u_alt)

                # shape: (batch, 2, ht/2**i, ht, wd) -> (batch, corr_channels-2, ht/2**i, ht, wd)
                a_v = attention2(sep_v)
                # shape: (batch, corr_channels-2, ht/2**i, ht, wd) -> (batch, corr_channels-2, ht/2**i, 1, ht, wd)
                a_v = a_v.unsqueeze(dim=3)
                a_v = a_v.softmax(dim=2)

                # shape:
                #       (batch, corr_channels-2,    ht/2**i,    1,          ht, wd)    attention
                #   *   (batch, 1,                  ht/2**i,    wd/2**i,    ht, wd)    4d correlation volume
                #   ->  (batch, corr_channels-2,    ht/2**i,    wd/2**i,    ht, wd)
                adaptive_corr_v = a_v * shaped_corr
                # shape:    (batch, corr_channels-2, ht/2**i, wd/2**i, ht, wd) 
                #   ->      (batch, corr_channels-2, wd/2**i, ht, wd)
                adaptive_corr_v = adaptive_corr_v.sum(dim=2)

                print(f"level {i}")
                print("sep before:")
                print(sep_v.shape)
                print(sep_u.shape)

                # shape: (batch, corr_channels, ht/2**i, ht, wd)
                sep_v = torch.cat((sep_v, adaptive_corr_u), dim=1)
                # shape: (batch, corr_channels, wd/2**i, ht, wd)
                sep_u = torch.cat((sep_u, adaptive_corr_v), dim=1)

                print("sep after:")
                print(sep_v.shape)
                print(sep_u.shape)

            # TODO: why upsample from reduced level resolution?
            # maybe they used the largest-level correlation volume to pair the attention with
            # also, upsampling is the only way the concatenation at the end is possible (otherwise, shape[2] would not match)
            # (batch, 2, wd/2**i, ht, wd) -> (batch, 2, wd, ht, wd)
            sep_u = F.interpolate(sep_u, [w2, h1, w1], mode='trilinear', align_corners=True)

            sep_u_lvls.append(sep_u)

            # (batch, 2, ht, ht, wd)
            sep_v = F.interpolate(sep_v, [h2, h1, w1], mode='trilinear', align_corners=True)

            sep_v_lvls.append(sep_v)
        
        # concatenate over all levels
        # liste -> (batch, 2*levels, wd, ht, wd)
        sep_u_lvls = torch.cat(sep_u_lvls, dim=1)
        # liste -> (batch, 2*levels, ht, ht, wd)
        sep_v_lvls = torch.cat(sep_v_lvls, dim=1)
        
        return sep_u_lvls, sep_v_lvls, adaptive_corr_u, adaptive_corr_v

    def separate_alt(self, attention_weights, attention_sum_method : Callable):
        batch, h1, w1, h2, w2 = self.shape

        sep_u_lvls = []
        sep_v_lvls = []

        adaptive_corr_u_lvls = []
        adaptive_corr_v_lvls = []

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

                # shape: (batch, 2, ht/2**i, ht, wd) -> (batch, corr_channels-2, ht/2**i, ht, wd)
                a_v = attention2(sep_v)
                # shape: (batch, corr_channels-2, ht/2**i, ht, wd) -> (batch, corr_channels-2, ht/2**i, 1, ht, wd)
                a_v = a_v.unsqueeze(dim=3)
                a_v = a_v.softmax(dim=2)

                adaptive_corr_u, adaptive_corr_v = attention_sum_method(corr, shaped_corr, a_u, a_v)

                # shape: (batch, corr_channels, ht/2**i, ht, wd)
                sep_v = torch.cat((sep_v, adaptive_corr_u), dim=1)
                # shape: (batch, corr_channels, wd/2**i, ht, wd)
                sep_u = torch.cat((sep_u, adaptive_corr_v), dim=1)

            # TODO: why upsample from reduced level resolution?
            # maybe they used the largest-level correlation volume to pair the attention with
            # also, upsampling is the only way the concatenation at the end is possible (otherwise, shape[2] would not match)
            # (batch, 2, wd/2**i, ht, wd) -> (batch, 2, wd, ht, wd)
            sep_u = F.interpolate(sep_u, [w2, h1, w1], mode='trilinear', align_corners=True)

            sep_u_lvls.append(sep_u)

            # (batch, 2, ht, ht, wd)
            sep_v = F.interpolate(sep_v, [h2, h1, w1], mode='trilinear', align_corners=True)

            sep_v_lvls.append(sep_v)

            adaptive_corr_u_lvls.append(adaptive_corr_u)
            adaptive_corr_v_lvls.append(adaptive_corr_v)
        
        # concatenate over all levels
        # liste -> (batch, 2*levels, wd, ht, wd)
        sep_u_lvls = torch.cat(sep_u_lvls, dim=1)
        # liste -> (batch, 2*levels, ht, ht, wd)
        sep_v_lvls = torch.cat(sep_v_lvls, dim=1)
        
        return sep_u_lvls, sep_v_lvls, adaptive_corr_u_lvls, adaptive_corr_v_lvls

    def apply_attention_loop(self, corr, shaped_corr, a_u, a_v):
        """ compute attention weighted sums for each direction

        Args:
            corr (torch.Tensor): correlation volume of shape (batch*ht*wd, 1, ht/2**i, wd/2**i)
            shaped_corr (torch.Tensor): correlation volume of shape (batch, 1, ht/2**i, wd/2**i, ht, wd)
            a_u (torch.Tensor): attention weights for C^A_u of shape (batch, corr_channels-2, 1, wd/2**i, ht, wd)
            a_v (torch.Tensor): attention weights for C^A_v of shape (batch, corr_channels-2, 1, ht/2**i, ht, wd)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: attention weighted, aggregated coorelation volume channels for C^A_u and C^A_v
        """

        batch, n_adaptive_channels, _, lvl_wd, h1, w1 = a_u.shape
        batch, n_adaptive_channels, lvl_ht, _, h1, w1 = a_v.shape
        
        # test if computation is same as paper
        # shape: (batch, corr_channels-2,    ht/2**i,    ht, wd)
        adaptive_corr_u = torch.zeros((batch, n_adaptive_channels, lvl_ht, h1, w1))
        for bn in range(batch):
            for channel in range(n_adaptive_channels):
                for i_idx in range(h1):
                    for j_idx in range(w1):
                        for u in range(lvl_ht):
                            adaptive_corr_u[bn,channel,u,i_idx,j_idx] += a_u[bn,channel,0,:,i_idx,j_idx].dot(shaped_corr[bn,0,u,:,i_idx,j_idx])

        
        # test if computation is same as paper
        # shape: (batch, corr_channels-2,    wd/2**i,    ht, wd)
        adaptive_corr_v = torch.zeros((batch, n_adaptive_channels, lvl_wd, h1, w1))
        for bn in range(batch):
            for channel in range(n_adaptive_channels):
                for i_idx in range(h1):
                    for j_idx in range(w1):
                        for v in range(lvl_wd):
                            adaptive_corr_v[bn,channel,v,i_idx,j_idx] += a_v[bn,channel,:,0,i_idx,j_idx].dot(shaped_corr[bn,0,:,v,i_idx,j_idx])
        
        return adaptive_corr_u, adaptive_corr_v

    def apply_attention_broadcasting(self, corr, shaped_corr, a_u, a_v):
        """ compute attention weighted sums for each direction

        Args:
            corr (torch.Tensor): correlation volume of shape (batch*ht*wd, 1, ht/2**i, wd/2**i)
            shaped_corr (torch.Tensor): correlation volume of shape (batch, 1, ht/2**i, wd/2**i, ht, wd)
            a_u (torch.Tensor): attention weights for C^A_u of shape (batch, corr_channels-2, 1, wd/2**i, ht, wd)
            a_v (torch.Tensor): attention weights for C^A_v of shape (batch, corr_channels-2, 1, ht/2**i, ht, wd)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: attention weighted, aggregated coorelation volume channels for C^A_u and C^A_v
        """
        
        # shape:
        #       (batch, corr_channels-2,    1,          wd/2**i, ht, wd)    attention
        #   *   (batch, 1,                  ht/2**i,    wd/2**i, ht, wd)    4d correlation volume
        #   ->  (batch, corr_channels-2,    ht/2**i,    wd/2**i, ht, wd)    
        adaptive_corr_u = a_u * shaped_corr

        # shape:    (batch, corr_channels-2,    ht/2**i,    wd/2**i,    ht, wd)
        #   ->      (batch, corr_channels-2,    ht/2**i,    ht, wd)
        adaptive_corr_u = adaptive_corr_u.sum(dim=3)

        # shape:
        #       (batch, corr_channels-2,    ht/2**i,    1,          ht, wd)    attention
        #   *   (batch, 1,                  ht/2**i,    wd/2**i,    ht, wd)    4d correlation volume
        #   ->  (batch, corr_channels-2,    ht/2**i,    wd/2**i,    ht, wd)
        adaptive_corr_v = a_v * shaped_corr
        # shape:    (batch, corr_channels-2, ht/2**i, wd/2**i, ht, wd) 
        #   ->      (batch, corr_channels-2, wd/2**i, ht, wd)
        adaptive_corr_v = adaptive_corr_v.sum(dim=2)

        return adaptive_corr_u, adaptive_corr_v

    def apply_attention_einsum(self, corr, shaped_corr, a_u, a_v):
        """ compute attention weighted sums for each direction

        Args:
            corr (torch.Tensor): correlation volume of shape (batch*ht*wd, 1, ht/2**i, wd/2**i)
            shaped_corr (torch.Tensor): correlation volume of shape (batch, 1, ht/2**i, wd/2**i, ht, wd)
            a_u (torch.Tensor): attention weights for C^A_u of shape (batch, corr_channels-2, 1, wd/2**i, ht, wd)
            a_v (torch.Tensor): attention weights for C^A_v of shape (batch, corr_channels-2, ht/2**i, 1, ht, wd)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: attention weighted, aggregated coorelation volume channels for C^A_u and C^A_v
        """
        
        adaptive_corr_u = torch.einsum('bcuvij,bcuvij->bcuij',a_u, shaped_corr)
        adaptive_corr_v = torch.einsum('bcuvij,bcuvij->bcvij',a_v, shaped_corr)

        return adaptive_corr_u, adaptive_corr_v

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_function(niter, fct, *args, **kwargs):
    timing1 = timedelta(0)
    for i in range(niter):
        start1 = datetime.now()
        corr1, corr2, adaptive_corr_u_lvls, adaptive_corr_v_lvls = fct(*args, **kwargs)
        stop1 = datetime.now()
        timing1 += (stop1-start1)
    timing1 = timing1 * (1/niter)
    print(timing1)

    return corr1, corr2, adaptive_corr_u_lvls, adaptive_corr_v_lvls

if __name__ == "__main__":

    K = 6

    attention1 = torch.nn.Conv3d(2, K-2, 3, padding=1)
    attention2 = torch.nn.Conv3d(2, K-2, 3, padding=1)

    print(count_parameters(attention1))

    corr_fn = CorrBlock(2, 16, 32, 4, 4)
    
    corr1, corr2, adaptive_corr_u_lvls1, adaptive_corr_v_lvls1 = benchmark_function(
        1000, corr_fn.separate_alt, (attention1, attention2), corr_fn.apply_attention_broadcasting)
    
    corr1, corr2, adaptive_corr_u_lvls2, adaptive_corr_v_lvls2 = benchmark_function(
        1000, corr_fn.separate_alt, (attention1, attention2), corr_fn.apply_attention_einsum)
    
    print("maximum per-value absolute error")
    for i in range(len(adaptive_corr_u_lvls1)):
        print(((adaptive_corr_u_lvls1[i] - adaptive_corr_u_lvls2[i]).abs()).max()/adaptive_corr_u_lvls1[i].flatten().shape[0])
        print(((adaptive_corr_v_lvls1[i] - adaptive_corr_v_lvls2[i]).abs()).max()/adaptive_corr_v_lvls1[i].flatten().shape[0])

    print("corr1/corr2 shape")
    print(corr1.shape)
    print(corr2.shape)