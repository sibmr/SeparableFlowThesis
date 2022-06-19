from torch.nn.modules.module import Module
import torch
import numpy as np
from torch.autograd import Variable
from ..functions import *
import torch.nn.functional as F

from ..functions.MemorySaver import ComputeMaxAvgFunction

class ComputeMaxAvgModule(Module):
    def __init__(self):
        super(ComputeMaxAvgModule, self).__init__()
    def forward(self, img1_features_l0, img2_features_lk):
        return ComputeMaxAvgFunction.apply(img1_features_l0, img2_features_lk)
