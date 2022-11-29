import torch
import torch.nn.functional as F
import torch.nn as nn
from libs.GANet.modules.GANet import NLFMax, NLFIter

class NLF(nn.Module):
    """ Aggregation Module for the 4D correlation volume
    """

    def __init__(self, in_channel=32):
        super(NLF, self).__init__()
        self.nlf = NLFIter()
    
    def forward(self, x, g):
        """ 4 directional aggregation of 4d correlation volume

        Args:
            x (torch.Tensor): 4d correlation volume
            g (torch.Tensor): guidance for 4d correlation volume (NLF filter weights)
        """
        N, D1, D2, H, W = x.shape

        # merge pixel height/width
        x = x.reshape(N, D1*D2, H, W).contiguous()
        
        # split guidance for each direction: down, up, right, left
        k1, k2, k3, k4 = torch.split(g, (5, 5, 5, 5), 1)
        
        # L1 normalize the guidance for each direction
        k1 = F.normalize(k1, p=1, dim=1)
        k2 = F.normalize(k2, p=1, dim=1)
        k3 = F.normalize(k3, p=1, dim=1)
        k4 = F.normalize(k4, p=1, dim=1)

        # apply nlf to 4d cost volume
        x = self.nlf(x, k1, k2, k3, k4)

        # reshape to separate pixel channels
        x = x.reshape(N, D1, D2, H, W)
        
        return x
 