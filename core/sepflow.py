import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, CorrBlock1D, AlternateCorrBlockSepflow
from cost_agg import CostAggregation
from guidance import Guidance
from gma import Attention
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class SepFlow(nn.Module):
    def __init__(self, args):
        super(SepFlow, self).__init__()
        self.args = args
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        if 'alternate_corr_backward' not in self.args:
            self.args.alternate_corr_backward = False
        
        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        self.guidance = Guidance(channels=256, no_4dcorr_aggregation=args.no_4d_agg)
        self.cost_agg1 = CostAggregation(in_channel=args.num_corr_channels*args.corr_levels)
        self.cost_agg2 = CostAggregation(in_channel=args.num_corr_channels*args.corr_levels)

        # if gma is used, create the gma attention network
        self.gma_attention_net = None
        if args.use_gma:
            self.gma_attention_net = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)
    
        # if there are more than two correlation channels, create networks for attention weights with K-2 channels
        if args.num_corr_channels > 2:
            self.attention1 = torch.nn.Conv3d(2, args.num_corr_channels-2, 3, padding=1)
            self.attention2 = torch.nn.Conv3d(2, args.num_corr_channels-2, 3, padding=1)
            self.attention_weights = (self.attention1, self.attention2)
        else:
            self.attention1 = None
            self.attention2 = None
            self.attention_weights = None

    def freeze_bn(self):
        count1, count2, count3 = 0, 0, 0
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                count1 += 1
                m.eval()
            if isinstance(m, nn.BatchNorm2d):
                count2 += 1
                m.eval()
            if isinstance(m, nn.BatchNorm3d):
                count3 += 1
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, upsample=True):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim
        
        # calculate per-eighth-pixel features
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)
        
        # cast to float
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # guidance is part of the Semi-Global-Aggregation Layer
        # returns 3 dictionaries for guidance of aggregation of 4d-uv, 3d-u and 3d-v correlation volume
        # guidance: 4 directions, with 5 weights each (summing to 1)
        guid, guid_u, guid_v = self.guidance(fmap1.detach(), image1)
        
        if self.args.alternate_corr or self.args.alternate_corr_backward:
            corr_fn = AlternateCorrBlockSepflow(fmap1, fmap2, guid, radius=self.args.corr_radius,
                                                support_backward=self.args.alternate_corr_backward)
        else:
            # correlation now seems to use guidance
            # corr_fn used for both 4d and 3d cost volume computation
            corr_fn = CorrBlock(fmap1, fmap2, guid, radius=self.args.corr_radius)

        # context features calculation:
        # hidden state initialization (net) + context features (inp)
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        
        gma_attention = None
        if self.gma_attention_net is not None:
            # query and key features calculated by attention network from context features
            gma_attention = self.gma_attention_net(inp)
        
        # calculated C_u and C_v
        # They consist of the max and avg of the u column/ v row
        # C_u: (batch, 2*levels, wd, ht, wd)
        # C_v: (batch, 2*levels, ht, ht, wd)
        corr1, corr2 = corr_fn(None, sep=True, attention_weights=self.attention_weights)
        
        # initializing the flow (like raft, zero-flow)
        # coords1 - coords0 = flow = zeros
        coords0, coords1 = self.initialize_flow(image1)

        # cost aggregation reduces corr1 and corr2 from (batch, K, ht, wd) to (batch, 1, ht, wd)
        if self.training:
            # is_ux = True identifies corr1 as C_u
            u0, u1, flow_u, corr1 = self.cost_agg1(corr1, guid_u, max_shift=384, is_ux=True)
            # is_ux = False identifies corr2 as C_v
            v0, v1, flow_v, corr2 = self.cost_agg2(corr2, guid_v, max_shift=384, is_ux=False)
            
            # motion-regressed inital flow
            flow_init = torch.cat((flow_u, flow_v), dim=1)
            
            # add all three parts of the motion regression of the initial flow
            # to the list for calculating the flow error loss
            flow_predictions = []
            flow_predictions.append(torch.cat((u0, v0), dim=1))
            flow_predictions.append(torch.cat((u1, v1), dim=1))
            flow_predictions.append(flow_init)
            
        else:
            # cost aggregation reduces corr1 and corr2 from (batch, K, ht, wd) to (batch, 1, ht, wd)
            flow_u, corr1 = self.cost_agg1(corr1, guid_u, max_shift=384, is_ux=True)
            flow_v, corr2 = self.cost_agg2(corr2, guid_v, max_shift=384, is_ux=False)

            # concatentate motion-regressed flow
            flow_init = torch.cat((flow_u, flow_v), dim=1)
        
        # downsample inital flow
        flow_init = F.interpolate(flow_init.detach()/8.0, [cnet.shape[2], cnet.shape[3]], mode='bilinear', align_corners=True)
        
        # create 1d correlation block from C^A_u and C^A_v
        corr1d_fn = CorrBlock1D(corr1, corr2, radius=self.args.corr_radius)
        
        # update coords1 with inital flow estimate
        coords1 = coords1 + flow_init
        
        # iterative optimization
        for itr in range(iters):
            coords1 = coords1.detach()
            
            # only index the 4d correlation volume if it is used
            if self.args.no_4d_corr:
                corr = None
            else:
                # index 4d correlation volume -> multi-scale correlation features
                corr = corr_fn(coords1) # index correlation volume
            
            # index the two 3d correlation volumes
            corr1, corr2 = corr1d_fn(coords1) # index correlation volume

            # calculate current flow
            flow = coords1 - coords0
            
            # apply update block: flow refinement
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, corr1, corr2, flow, gma_attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            # add flow refinement step to list for loss computation 
            if self.training:
                flow_predictions.append(flow_up)

        if self.training:
            # shape: (batch, 2, HT, WD)
            return flow_predictions
        else:
            return coords1 - coords0, flow_up
            
