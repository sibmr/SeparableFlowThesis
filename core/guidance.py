import torch
import torch.nn as nn

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
        # both together: (batch, self.channels/4, ht, wd) -> (batch, self.channels/2, ht//2, wd//2)
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
        
        


    def forward(self, fea, img):
        """ calculate guidance for 4D and 3D correspondence volume aggregation

        Args:
            fea (torch.Tensor): image1 features with shape (batch, fdim, ht, wd)
            img (torch.Tensor): image1 with shape (batch, 3, ht, wd)

        Returns:
            Tuple[Optional[torch.Tensor], dict, dict]: guidance for 4d/3d correlation volume calculation
                - the Tensor is used for 4D correlation volume aggregation
                - the dictionaries are used for C_u, C_v 3d correlation volume aggregation
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
        # finally: (batch, 2*self.wsize, ht//2, wd//2) for sg11 and sg12 each
        sg11 = self.weight_sg11(x)
        sg11_u, sg11_v = torch.split(sg11, (self.wsize, self.wsize), dim=1)
        sg12 = self.weight_sg12(x)
        sg12_u, sg12_v = torch.split(sg12, (self.wsize, self.wsize), dim=1)
        
        # guidance with shapes 
        #   (batch, self.wsize, ht,    wd   ) for sg1, sg2, sg3
        #   (batch, self.wsize, ht//2, wd//2) for sg11, sg12
        # stored in dict
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