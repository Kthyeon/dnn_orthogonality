'''MicroNet

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
import numpy as np


class HSwish(nn.Module):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    inplace : bool
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace
        self.relu = nn.ReLU6(inplace = self.inplace)

    def forward(self, x):
        return x * self.relu(x + 3.0) / 6.0

class MicroBlock(nn.Module):
    '''expand + depthwise + pointwise
    Activation : ReLU or HSwish
    
    '''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(MicroBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        planes = int(expansion * in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.01)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
       
        
        # self.conv2 = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.01)
        
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes, momentum=0.01)
        
        self.act1 = HSwish()
        self.act2 = HSwish()
        self.act_se = HSwish()
                    
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes, momentum=0.01)
            )
        
        # SE layers
        self.avg_se = nn.AdaptiveAvgPool2d(1)
        number = int(out_planes*0.25)
        self.fc1 = nn.Conv2d(out_planes, number, kernel_size=1)
        self.fc2 = nn.Conv2d(number, out_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Squeeze-Excitation
        w = self.avg_se(out)
        w = self.act_se(self.fc1(w))
        w = self.sigmoid(self.fc2(w))

        out = out * w + self.shortcut(x) if self.stride==1 else out
        return out
    
    def norm_dif(self, x):
        norm_loss = 0.
        out = self.act1(self.bn1(self.conv1(x)))
        out_norm = torch.norm(out.reshape(out.shape[0],-1), dim=1)
        x_norm = torch.norm(x.reshape(x.shape[0],-1), dim=1)
        norm_loss += torch.sum(torch.abs(out_norm-x_norm))
        x1=out
        
        out = self.act2(self.bn2(self.conv2(x1)))
        out_norm = torch.norm(out.reshape(out.shape[0],-1), dim=1)
        x_norm = torch.norm(x1.reshape(x1.shape[0],-1), dim=1)
        norm_loss += torch.sum(torch.abs(out_norm-x_norm))
        x1=out
        
        out = self.bn3(self.conv3(x1))
        out_norm = torch.norm(out.reshape(out.shape[0],-1), dim=1)
        x_norm = torch.norm(x1.reshape(x1.shape[0],-1), dim=1)
        norm_loss += torch.sum(torch.abs(out_norm-x_norm))
        
        # Squeeze-Excitation
        w = self.avg_se(out)
        w = self.act_se(self.fc1(w))
        w = self.sigmoid(self.fc2(w))

        out = out * w + self.shortcut(x) if self.stride==1 else out
        out_norm = torch.norm(out.reshape(out.shape[0],-1), dim=1)
        x_norm = torch.norm(x.reshape(x.shape[0],-1), dim=1)
        norm_loss += torch.sum(torch.abs(out_norm-x_norm))
        
        return norm_loss
        
class MicroNet(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    def __init__(self, opt='both', init='xavier', ver = 'ver2', num_classes=100, wide_factor = 1, depth_factor =1, expansion = 1.):
        super(MicroNet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        # opt: option that orthogonalizes the pointwise conv in either expension or reduction or both.
        # opt in ['exp', 'rec', 'both']
        # init in ['xavier', 'kaiming', 'ort', 'z_ort']
        '''
        wide_factor: widening ratio of width
        depth_factor: expanding ratio of depth
        '''
        if ver == 'ver2':
            self.cfg = [[1, 20, 2, 1],
                        [1, 36, 1, 2],
                        [1, 36, 1, 1],
                        [1, 56, 3, 1],
                        [1, 80, 1, 2],
                        [1, 80, 4, 1],
                        [1, 88, 1, 2],
                        [1, 96, 2, 1],
                        [1, 114, 1, 1]]
        else:
            self.cfg = [[1, 16, 2, 1],
                        [1, 32, 1, 2],
                        [1, 32, 1, 1],
                        [1, 48, 3, 1],
                        [1, 72, 1, 2],
                        [1, 72, 4, 1],
                        [1, 80, 1, 2],
                        [1, 88, 2, 1],
                        [1, 106, 1, 1]]
        
        for number in range(len(self.cfg)):
            self.cfg[number][0] = int(self.cfg[number][0] * expansion)

        #init option
        self.init = init
        self.opt = opt
        
        #reconstruct structure config
        self.change_cfg(wide_factor, depth_factor)
        
        #construct network
        self.input_channel = 32
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, self.input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(self.input_channel, momentum=0.01)
        self.blocks = self._make_layers(in_planes=self.input_channel)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.cfg[-1][1], self.num_classes)
        self.stem_act = HSwish()
        
        #initialize the parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        # This is for the initialization of the parameters in the network. For both, beta and gamma in batchnorm is set to 0 and 1, respectively.
        # Xavier: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        # Kaiming: https://arxiv.org/pdf/1502.01852.pdf
        if self.init == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    m.bias.data.zero_()
        elif self.init == 'kaiming':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    m.bias.data.zero_()
        elif self.init == 'ort':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.weight.shape[2] == 1:
                        nn.init.orthogonal_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    m.bias.data.zero_()
        elif self.init == 'z_ort':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.weight.shape[2] == 1:
                        if self.opt == 'exp' and m.weight.shape[0] < m.weight.shape[1]:
                            continue
                        elif self.opt == 'rec' and m.weight.shape[0] > m.weight.shape[1]:
                            continue
                        nn.init.orthogonal_(m.weight)
                    elif m.weight.shape[2] ==3 and m.weight.shape[1] == 1:
                        m.weight.data.zero_()
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    m.bias.data.zero_() 
                
                        
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(MicroBlock(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)
    
    def change_cfg(self, wide_factor, depth_factor):
        for i in range(len(self.cfg)):
            self.cfg[i][1] = int(self.cfg[i][1] * wide_factor)
            if self.cfg[i][3] ==1:
                self.cfg[i][2] = int(self.cfg[i][2] * depth_factor)
    
    def forward(self, x):
        #stem
        out = self.stem_act(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.linear(self.dropout(out))
            
        return out
    
    def make_norm_dif(self, x, loc = [True, True]):
        # loc is the location vector whether penalize norm or not.
        # loc[0]: stem
        # loc[1]: fully
        norm_loss = 0.
        num = 0
        out = self.stem_act(self.bn1(self.conv1(x)))
        if loc[0]:
            out_norm = torch.norm(out.reshape(out.shape[0],-1), dim=1)
            x_norm = torch.norm(x.reshape(x.shape[0],-1), dim=1)
            norm_loss += torch.sum(torch.abs(out_norm-x_norm))
            num += 1
        
        x = out
        for layer in self.blocks:
            #if layer.downsample:
            #    x = layer(x)
            #    continue
            out = layer(x)
            norm_loss += layer.norm_dif(x)
            x = out
            num += 4
        
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        out = self.linear(self.dropout(x))
        if loc[1]:
            out_norm = torch.norm(out.reshape(out.shape[0],-1), dim=1)
            x_norm = torch.norm(x.reshape(x.shape[0],-1), dim=1)
            norm_loss += torch.sum(torch.abs(out_norm-x_norm))
            num += 1
        
        return norm_loss / num / x.shape[0]
    
    
    def extract_feature(self, x):
        # To analyze the behavior of layerwise response.
        # feature_list: layerwise output whose channel size is 1. (kind of activatoin map) (shape: NxNx1)
        # map_list: avgpool of each instance (shape: 1x1xN)
        # norm_list: norm of each instance (2-norm)
        feature_list = []
        map_list = []
        norm_list = []

        out = self.stem_act(self.bn1(self.conv1(x)))
        for block in self.blocks:
            tmp_f = []
            tmp_m = []
            tmp_n = []
            # input
            tmp_f.append(F.adaptive_avg_pool2d(F.adaptive_avg_pool2d(out,2), 3))
            tmp_m.append(F.adaptive_avg_pool2d(out, 1))
            tmp_n.append(torch.norm(out,2))

            out1 = block.act1(block.bn1(block.conv1(out1)))
            # after expand conv
            tmp_f.append(F.adaptive_avg_pool2d(F.adaptive_avg_pool2d(out1,2), 3))
            tmp_m.append(F.adaptive_avg_pool2d(out1, 1))
            tmp_n.append(torch.norm(out1,2))

            out1 = block.act2(block.bn2(block.conv2(out1)))
            #after depthwise and se block
            tmp_f.append(F.adaptive_avg_pool2d(F.adaptive_avg_pool2d(out1,2), 3))
            tmp_m.append(F.adaptive_avg_pool2d(out1, 1))
            tmp_n.append(torch.norm(out1,2))

            out = block(out)
            #after pointwise and skip connection
            tmp_f.append(F.adaptive_avg_pool2d(F.adaptive_avg_pool2d(out,2), 3))
            tmp_m.append(F.adaptive_avg_pool2d(out, 1))
            tmp_n.append(torch.norm(out,2))

            # append
            feature_list.append(tmp_f)
            map_list.append(tmp_m)
            norm_list.append(tmp_n)

        
        return feature_list, map_list, norm_list
    
    def norm_change(self,x):
        norm_list = []
        
        out = self.stem_act(self.bn1(self.conv1(x)))
        norm_list.append(torch.norm(x).item()-torch.norm(out).item())
        for block in self.blocks:
            out1 = block(out)
            norm_list.append(torch.norm(out).item()-torch.norm(out1).item())
            out = out1
        
        return norm_list