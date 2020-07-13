'''Mobile V2

Option
1. Activation : ReLU, HSwish
2. Squeeze - and - Excitation ratio

    No pruning version
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
import math
import torchvision


class MobileV2(nn.Module):
    def __init__(self, opt='both', init='xavier', pre_trained=False, num_classes = 1000):
        # opt: option that orthogonalizes the pointwise conv in either expension or reduction or both.
        # opt in ['exp', 'rec', 'both']
        # init in ['xavier', 'kaiming', 'ort', 'z_ort']
        super(MobileV2, self).__init__()
        self.init = init
        self.opt = opt
        if pre_trained:
            self.net = torchvision.models.mobilenet_v2(pretrained=True, progress=True, num_classes = num_classes)
        else:
            self.net = torchvision.models.mobilenet_v2(pretrained=False, progress=True, num_classes = num_classes)

        #initialize the parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        # This is for the initialization of the parameters in the network. For both, beta and gamma in batchnorm is set to 0 and 1, respectively.
        # Xavier: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        # Kaiming: https://arxiv.org/pdf/1502.01852.pdf
        if self.init == 'xavier':
            for m in self.net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
        elif self.init == 'kaiming':
            for m in self.net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
        elif self.init == 'ort':
            for m in self.net.modules():
                if isinstance(m, nn.Conv2d):
                    if m.weight.shape[2] == 1:
                        nn.init.orthogonal_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init>constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.constant_(m.bias, 0)
        elif self.init == 'z_ort':
            for m in self.net.modules():
                if isinstance(m, nn.Conv2d):
                    if m.weight.shape[2] == 1:
                        if self.opt == 'exp' and m.weight.shape[0] < m.weight.shape[1]:
                            continue
                        elif self.opt == 'rec' and m.weight.shape[0] > m.weight.shape[1]:
                            continue
                        nn.init.orthogonal_(m.weight)
                    elif m.weight.shape[2] ==3 and m.weight.shape[1] == 1:
                        nn.init.constant_(m.weight, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init>constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.constant_(m.bias, 0)   
        
    def forward(self, x):
        out = self.net(x)
            
        return out

    

