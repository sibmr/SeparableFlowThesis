import torch
import torch.nn.functional as F

class CorrBlock:
    
    def __init__(self, batch, ht, wd, num_levels, radius):
        
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = torch.ones((batch, ht, wd, ht, wd))

        batch, h1, w1, h2, w2 = corr.shape
        self.shape = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, h2, w2)
        
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
        
        return sep_u_lvls, sep_v_lvls

if __name__ == "__main__":

    K = 6

    attention1 = torch.nn.Conv3d(2, K-2, 3, padding=1)
    attention2 = torch.nn.Conv3d(2, K-2, 3, padding=1)

    corr_fn = CorrBlock(2, 16, 32, 4, 4)

    corr1, corr2 = corr_fn.separate((attention1, attention2))

    print("corr1/corr2 shape")
    print(corr1.shape)
    print(corr2.shape)