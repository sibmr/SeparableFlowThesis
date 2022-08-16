import torch
import torch.nn as nn
import torch.nn.functional as F

from gma import Aggregate

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    """ computes motion features """
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        
        # number of correlation features per pixel in 4d correlation volume
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        
        # number of correlation features per pixel in 3d correlation volumes
        cor1_planes = args.corr_levels * (2*args.corr_radius + 1)
        
        # inital, channel-combining layers with kernel size 1
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc11 = nn.Conv2d(cor1_planes, 64, 1, padding=0)
        self.convc12 = nn.Conv2d(cor1_planes, 64, 1, padding=0)
        
        # intermediate, spatial/channel-combining layers with kernel size 3
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convc21 = nn.Conv2d(64, 64, 3, padding=1)
        self.convc22 = nn.Conv2d(64, 64, 3, padding=1)
        
        # flow processing layers
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        
        # final convolution, combining correlation volumes and flow
        # (batch, 64+192+64*2, ht, wd) -> (batch, 128-2, ht, wd)
        self.conv = nn.Conv2d(64+192+64*2, 128-2, 3, padding=1)

    def forward(self, flow, corr, corr1, corr2):
        # process 4d correlation volume
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        # process 3d correlation volume 1
        cor1 = F.relu(self.convc11(corr1))
        cor1 = F.relu(self.convc21(cor1))
        # process 3d correlation volume 2
        cor2 = F.relu(self.convc12(corr2))
        cor2 = F.relu(self.convc22(cor2))
        # process current flow
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        # combine flow and correlation volume in final layer
        cor_flo = torch.cat([cor, cor1, cor2, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        
        # preserve original flow as motion feature by adding it to the output
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoderNo4dCorr(nn.Module):
    """ computes motion features without using 4d correlation volume"""
    
    def __init__(self, args):
        super(BasicMotionEncoderNo4dCorr, self).__init__()
        
        # number of correlation features per pixel in 3d correlation volumes
        cor1_planes = args.corr_levels * (2*args.corr_radius + 1)
        
        # inital, channel-combining layers with kernel size 1
        self.convc11 = nn.Conv2d(cor1_planes, 64, 1, padding=0)
        self.convc12 = nn.Conv2d(cor1_planes, 64, 1, padding=0)
        
        # intermediate, spatial/channel-combining layers with kernel size 3
        self.convc21 = nn.Conv2d(64, 64, 3, padding=1)
        self.convc22 = nn.Conv2d(64, 64, 3, padding=1)
        
        # flow processing layers
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        
        # final convolution, combining correlation volumes and flow
        # (batch, 64+192+64*2, ht, wd) -> (batch, 128-2, ht, wd)
        self.conv = nn.Conv2d(64 + 64*2, 128-2, 3, padding=1)

    def forward(self, flow, corr1, corr2):
        # process 3d correlation volume 1
        cor1 = F.relu(self.convc11(corr1))
        cor1 = F.relu(self.convc21(cor1))
        # process 3d correlation volume 2
        cor2 = F.relu(self.convc12(corr2))
        cor2 = F.relu(self.convc22(cor2))
        # process current flow
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        # combine flow and correlation volume in final layer
        cor_flo = torch.cat([cor1, cor2, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        
        # preserve original flow as motion feature by adding it to the output
        return torch.cat([out,   flow], dim=1)

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args

        if self.args.no_4d_corr:
            self.encoder = BasicMotionEncoderNo4dCorr(args)
        else:
            self.encoder = BasicMotionEncoder(args)
        
        self.gma_aggregator_net = None
        gru_input_dim = 128+hidden_dim
        if self.args.use_gma:
            self.gma_aggregator_net = Aggregate(args=self.args, dim=128, dim_head=128, heads=self.args.num_heads)
            gru_input_dim = 128+hidden_dim+hidden_dim

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=gru_input_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, corr1, corr2, flow, gma_attention, upsample=True):
        
        # motion features include flow and a 4d and 3d motion representation
        # compared to raft, includes additional corr1 and corr2 volume
        if self.args.no_4d_corr:
            motion_features = self.encoder(flow, corr1, corr2)
        else:
            motion_features = self.encoder(flow, corr, corr1, corr2)
            
        if self.args.use_gma:
            # context, motion and aggregated global motion features
            motion_features_global = self.gma_aggregator_net(gma_attention, motion_features)
            inp = torch.cat([inp, motion_features, motion_features_global], dim=1)
        else:
            # same combining context and motion features, same as raft
            inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow



