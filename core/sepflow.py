import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, CorrBlock1D, AlternateCorrBlockSepflow
from cost_agg import CostAggregation
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
class Guidance(nn.Module):
    def __init__(self, channels=32, no_4dcorr_aggregation=False):
        """ create the guidance subnetwork

        Args:
            channels (int, optional):   Largest channel number used by the initial hourglass network.
                                        Defaults to 32.
            no_4dcorr_aggregation (bool, optional): Whether to compute weights for 4d correlation volume aggregation. 
                                                    Defaults to False.
        """
        super(Guidance, self).__init__()
        
        # norm + activation for feature input
        self.bn_relu = nn.Sequential(nn.InstanceNorm2d(channels),
                                     nn.ReLU(inplace=True))
        
        # first convolution stack
        # reduces resolution by 8, like feature/context network
        # input:            (batch, 3           , HT    , WD    )
        # after 1. conv:    (batch, 16          , HT    , WD    )
        # after 2. conv:    (batch, channels/4  , HT/2  , WD/2  )
        # after 3. conv:    (batch, channels/2  , HT/4  , WD/4  )
        # after 4. conv:    (batch, channels    , HT/8  , WD/8  )
        # same as:          (batch, channels    , ht    , wd    )
        self.conv0 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1),
                                   nn.InstanceNorm2d(16),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, int(channels/4), kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(int(channels/4)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(int(channels/4), int(channels/2), kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(int(channels/2)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(int(channels/2), channels, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(channels),
                                   nn.ReLU(inplace=True))
        
        inner_channels = channels // 4
        self.wsize = 20
        
        # reduce number channels for concatenated processed image and feature tensor
        # (batch, 2*self.channels, ht, wd) -> (batch, self.channels/4, ht, wd)
        self.conv1 = nn.Sequential(nn.Conv2d(channels*2, inner_channels, kernel_size=3, padding=1),
                                   nn.InstanceNorm2d(inner_channels),
                                   nn.ReLU(inplace=True))
        
        # standard 2-layer convolutional net with instance norm
        # does not change shape: (batch, channels/4, ht, wd) -> (batch, channels/4, ht, wd)
        self.conv2 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
                                   nn.InstanceNorm2d(inner_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1),
                                   nn.InstanceNorm2d(inner_channels),
                                   nn.ReLU(inplace=True))
        
        # same as conv2
        self.conv3 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
                                   nn.InstanceNorm2d(inner_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1),
                                   nn.InstanceNorm2d(inner_channels),
                                   nn.ReLU(inplace=True))
        
        # last two processing blocks
        # combined in additive residual style
        # both together: (batch, self.channels/4, ht, wd) -> (batch, self.channels/2, ht, wd)
        self.conv11 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels*2, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(inner_channels*2),
                                   nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(nn.Conv2d(inner_channels*2, inner_channels*2, kernel_size=3, stride=1, padding=1),
                                   nn.InstanceNorm2d(inner_channels*2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inner_channels*2, inner_channels*2, kernel_size=3, stride=1, padding=1),
                                   nn.InstanceNorm2d(inner_channels*2),
                                   nn.ReLU(inplace=True))
        
        # no 4d correlation volume aggregation if argument is true
        if no_4dcorr_aggregation:
            self.weights = None
        # by default, use 4d correlation volume aggregation
        else:
            # weights for 4d correlation volume guidance
            # (batch, self.channels/4, ht, wd) -> (batch, self.wsize, ht, wd)
            self.weights = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
                                            nn.InstanceNorm2d(inner_channels),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(inner_channels, self.wsize, kernel_size=3, stride=1, padding=1))
        
        # weight_sg1, weight_sg2, weight_sg3: identical
        # two convolutions, where second has un-activated regression output to module
        # shape: (batch, self.channels/4, ht, wd) -> (batch, 2*self.wsize, ht, wd)
        self.weight_sg1 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(inner_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inner_channels, self.wsize*2, kernel_size=3, stride=1, padding=1))
        # same as weight_sg1 and weight_sg3
        self.weight_sg2 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(inner_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inner_channels, self.wsize*2, kernel_size=3, stride=1, padding=1))
        
        # weight_sg11 and weight_sg12: identical
        # shape: (batch, self.channels/2, ht, wd) -> (batch, 2*self.wsize, ht, wd)
        self.weight_sg11 = nn.Sequential(nn.Conv2d(inner_channels*2, inner_channels*2, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(inner_channels*2),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inner_channels*2, self.wsize*2, kernel_size=3, stride=1, padding=1))
        self.weight_sg12 = nn.Sequential(nn.Conv2d(inner_channels*2, inner_channels*2, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(inner_channels*2),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inner_channels*2, self.wsize*2, kernel_size=3, stride=1, padding=1))
        
        # same as weight_sg1 and weight_sg2
        self.weight_sg3 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(inner_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(inner_channels, self.wsize*2, kernel_size=3, stride=1, padding=1))
        
        
        #self.getweights = nn.Sequential(GetFilters(radius=1),
        #                                nn.Conv2d(9, 20, kernel_size=1, stride=1, padding=0, bias=False))



    def forward(self, fea, img):
        """ calculate guidance for 4D and 3D correspondence volume aggregation

        Args:
            fea (torch.Tensor): image1 features with shape (batch, fdim, ht, wd)
            img (torch.Tensor): image1 with shape (batch, 3, ht, wd)

        Returns:
            Tuple[Optional[torch.Tensor], dict, dict]: guidance for 4d/3d correlation volume calculation
                the Tensor is used for correlation volume aggregation
                the dictionaries are used for C_u, C_v 3d correlation volume aggregation
                dictionaries contain keys sg1,sg2,sg3,sg11,sg12 for the 5 different aggregation steps
                each one-eighth-pixel has 20 weights that are split as follows:
                    4 directions, with 5 weights each for aggregation
                the weights for 4d cost volume aggregation may be None
                if this is the case, the guidance parameters for the 4d volume are also None
        """
        
        # reduce img resolution from full to 1/8, adding channels
        x = self.conv0(img)
        
        # adding feature channels to image
        # (batch, self.channels, ht, wd) (batch, fdim, ht, wd) -> (batch, self.channels+fdim, ht, wd)
        # needs assert(fdim == self.channels) since resulting shape needs to be (batch, 2*self.channels, ht, wd)
        x = torch.cat((self.bn_relu(fea), x), 1)
        
        # (batch, 2*self.channels, ht, wd) -> (batch, self.channels/4, ht, wd)
        x = self.conv1(x)
        rem = x

        # residual style addition of previous stage image
        x = self.conv2(x) + rem
        rem = x

        # use 4d correlation volume aggregation if weights network exists
        if self.weights is None:
            guid = None
        else:
            # guidance for 4d correlation volume computed at an early stage
            # shape: (batch, self.wsize, ht, wd)
            guid = self.weights(x)
        
        # another residual style block
        x = self.conv3(x) + rem
        
        # sg1, sg2, sg3 provide first three parts of 3D correlation volume guidance
        # in both u and v direction
        # so far: (batch, 3*self.wsize, ht, wd) for guid_u, guid_v each
        sg1 = self.weight_sg1(x)
        sg1_u, sg1_v = torch.split(sg1, (self.wsize, self.wsize), dim=1)
        sg2 = self.weight_sg2(x)
        sg2_u, sg2_v = torch.split(sg2, (self.wsize, self.wsize), dim=1)
        sg3 = self.weight_sg3(x)
        sg3_u, sg3_v = torch.split(sg3, (self.wsize, self.wsize), dim=1)
        
        # residual block for further processing
        x = self.conv11(x)
        rem = x 
        x = self.conv12(x) + rem
        
        # sg11 and sg12 calculated using strongest-processed input
        # provide last two parts of 3D correlation volume guidance
        # finally: (batch, 5*self.wsize, ht, wd) for guid_u, guid_v each
        sg11 = self.weight_sg11(x)
        sg11_u, sg11_v = torch.split(sg11, (self.wsize, self.wsize), dim=1)
        sg12 = self.weight_sg12(x)
        sg12_u, sg12_v = torch.split(sg12, (self.wsize, self.wsize), dim=1)
        
        # guidance with shape (batch, 5*self.wsize, ht, wd) stored in dict
        # as tensors of shape each (batch, self.wsize, ht, wd)
        guid_u = dict([('sg1', sg1_u),
                       ('sg2', sg2_u),
                       ('sg3', sg3_u),
                       ('sg11', sg11_u),
                       ('sg12', sg12_u)])
        
        guid_v = dict([('sg1', sg1_v),
                       ('sg2', sg2_v),
                       ('sg3', sg3_v),
                       ('sg11', sg11_v),
                       ('sg12', sg12_v)])
        
        return guid, guid_u, guid_v 


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
                #print(m)
                m.eval()
        #print(count1, count2, count3)
                #print(m)

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
        # hidden state initialization + context features
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        
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
            flow_u, corr1 = self.cost_agg1(corr1, guid_u, max_shift=384, is_ux=True)
            flow_v, corr2 = self.cost_agg2(corr2, guid_v, max_shift=384, is_ux=False)
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
                net, up_mask, delta_flow = self.update_block(net, inp, corr, corr1, corr2, flow)

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
            
