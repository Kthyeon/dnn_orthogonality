import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import cuda
import numpy as np
import torch

'''
<Reference>

ResNet for Cifar-100 is from:
[1] Clova AI Research, GitHub repository, https://github.com/clovaai/overhaul-distillation
'''

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, batchnorm = True, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.batchnorm = batchnorm
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        
        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out
    
    def norm_dif(self, x):
        norm_loss = []
        x1 = F.relu(x)
        residual = x1

        out = self.conv1(x1)
        x_norm = torch.sum(torch.square(x1.reshape(x1.shape[0],-1)), dim=1)
        if self.batchnorm:
            out = self.bn1(out)
        out_norm = torch.sum(torch.square(out.reshape(out.shape[0],-1)), dim=1)
        norm_loss.append((torch.sum(torch.abs(out_norm/x_norm)) / x_norm.shape[0]).item())
        out = self.relu(out)
        
        

        out1 = self.conv2(out)
        x_norm = torch.sum(torch.square(out.reshape(out.shape[0],-1)), dim=1)
        if self.batchnorm:
            out1 = self.bn2(out1)
        out_norm = torch.sum(torch.square(out1.reshape(out1.shape[0],-1)), dim=1)
        norm_loss.append((torch.sum(torch.abs(out_norm/x_norm)) / x_norm.shape[0]).item())


        if self.downsample is not None:
            residual = self.downsample(x1)

        out1 += residual

        #out_norm = torch.square(torch.norm(out1.reshape(out.shape[0],-1), dim=1))
        #x_norm = torch.square(torch.norm(x.reshape(x.shape[0],-1), dim=1))
        #norm_loss += torch.sum(torch.abs(out_norm-x_norm))/x_norm.shape[0]

        return norm_loss


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, batchnorm = True, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.batchnorm = batchnorm
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batchnorm:
            out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out
    
    def norm_dif(self, x):
        epsilon = 1e-5
        norm_loss = []
        
        x1 = F.relu(x)
        residual = x1

        out = self.conv1(x1)
        x_norm = torch.sum(torch.square(x1.reshape(x1.shape[0],-1)), dim=1)
        out_norm = torch.sum(torch.square(out.reshape(out.shape[0],-1)), dim=1)
        norm_loss.append((torch.sum(torch.abs(out_norm/x_norm)) / x_norm.shape[0]).item())
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu(out)

        
        out1 = self.conv2(out)
        x_norm = torch.sum(torch.square(out.reshape(out.shape[0],-1)), dim=1)
        out_norm = torch.sum(torch.square(out1.reshape(out1.shape[0],-1)), dim=1)
        norm_loss.append((torch.sum(torch.abs(out_norm/x_norm)) / x_norm.shape[0]).item())
        if self.batchnorm:
            out1 = self.bn2(out1)
        out1 = self.relu(out1)
        
        out = self.conv3(out1)
        x_norm = torch.sum(torch.square(out1.reshape(out1.shape[0],-1)), dim=1)
        out_norm = torch.sum(torch.square(out.clone().reshape(out.shape[0],-1)), dim=1)
        norm_loss.append((torch.sum(torch.abs(out_norm/x_norm)) / x_norm.shape[0]).item())
        
        if self.batchnorm:
            out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x1)

        out += residual
        #out_norm = torch.sum(torch.square(out.reshape(out.shape[0],-1)), dim=1)
        #x_norm = torch.sum(torch.square(x.clone().reshape(x.shape[0],-1)), dim=1)
        #norm_loss += torch.sum(torch.abs(out_norm-x_norm))/x_norm.shape[0]
        
        return norm_loss
    

class ResNet(nn.Module):
    def __init__(self, width, depth, num_classes, batchnorm, opt = 'both', init = 'xavier', bottleneck=False):
        super(ResNet, self).__init__()
        self.inplanes = 16 * width
        if bottleneck == True:
            n = int((depth - 2) / 9)
            block = Bottleneck
        else:
            n = int((depth - 2) / 6)
            block = BasicBlock
        
        #init option
        self.init = init
        self.opt = opt
        self.batchnorm = batchnorm
        
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16* width, n)
        self.layer2 = self._make_layer(block, 32* width, n, stride=2)
        self.layer3 = self._make_layer(block, 64* width, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * width * block.expansion, num_classes)

        self.reset_parameters()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.batchnorm:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                )

        layers = []
        layers.append(block(inplanes = self.inplanes, planes=planes, stride=stride, downsample = downsample, batchnorm=self.batchnorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def make_norm_dif(self, x, loc = [True, True], down = False):
        # loc is the location vector whether penalize norm or not.
        # loc[0]: stem
        # loc[1]: fully
        norm_loss = []
        num = 0
        x_norm = torch.sum(torch.square(x.reshape(x.shape[0],-1)), dim=1)
        out = self.conv1(x)
        if loc[0]:
            out_norm = torch.sum(torch.square(out.reshape(out.shape[0],-1)), dim=1)
            norm_loss.append((torch.sum(torch.abs(out_norm/x_norm)) / x_norm.shape[0]).item())
        
        if self.batchnorm:
            out = self.bn1(out)
        
        x = out
        for layer in self.layer1:
            if layer.downsample and down:
                x = layer(x)
                continue
            out = layer(x)
            norm_loss += layer.norm_dif(x)
            x = out
            num += 3
            
        for layer in self.layer2:
            if layer.downsample and down:
                x = layer(x)
                continue
            out = layer(x)
            norm_loss += layer.norm_dif(x)
            x = out
            num += 3
            
        for layer in self.layer3:
            if layer.downsample and down:
                x = layer(x)
                continue
            out = layer(x)
            norm_loss += layer.norm_dif(x)
            x = out
            num += 3
        
        x = F.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        out = self.fc(x)
        if loc[1]:
            out_norm = torch.sum(torch.square(out.reshape(out.shape[0],-1)), dim=1)
            x_norm = torch.sum(torch.square(x.reshape(x.shape[0],-1)), dim=1)
            norm_loss.append((torch.sum(torch.abs(out_norm/x_norm)) / x_norm.shape[0]).item())

        
        return norm_loss
    
    def forward(self, x):

        x = self.conv1(x)
        if self.batchnorm:
            x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            print('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]

    def get_channel_num(self):

        return [16, 32, 64]

    def extract_feature(self, x, preReLU=False):

        x = self.conv1(x)
        x = self.bn1(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        x = F.relu(feat3)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)

        return [feat1, feat2, feat3], out
    
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
        elif self.init =='delta':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.dirac_(m.weight)
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
                        nn.init.constant_(m.weight, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    m.bias.data.zero_()

def ResNet50_4():
    
    return ResNet(opt = 'both', init = 'ortho', num_classes = 10, batchnorm = True, width = 4, depth = 50, bottleneck= False)


def conv_delta_orthogonal_(tensor, gain=1.):
    r"""Initializer that generates a delta orthogonal kernel for ConvNets.
    The shape of the tensor must have length 3, 4 or 5. The number of input
    filters must not exceed the number of output filters. The center pixels of the
    tensor form an orthogonal matrix. Other pixels are set to be zero. See
    algorithm 2 in [Xiao et al., 2018]: https://arxiv.org/abs/1806.05393
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`3 \leq n \leq 5`
        gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
    Examples:
        >>> w = torch.empty(5, 4, 3, 3)
        >>> nn.init.conv_delta_orthogonal_(w)
    """
    if tensor.ndimension() < 3 or tensor.ndimension() > 5:
        raise ValueError("The tensor to initialize must be at least "
                       "three-dimensional and at most five-dimensional")
    
    if tensor.size(1) > tensor.size(0):
        raise ValueError("In_channels cannot be greater than out_channels.")
    
    # Generate a random matrix
    a = tensor.new(tensor.size(0), tensor.size(0)).normal_(0, 1)
    # Compute the qr factorization
    q, r = torch.qr(a)
    # Make Q uniform
    d = torch.diag(r, 0)
    q *= d.sign()
    q = q[:, :tensor.size(1)]
    with torch.no_grad():
        tensor.zero_()
        if tensor.ndimension() == 3:
            tensor[:, :, (tensor.size(2)-1)//2] = q
        elif tensor.ndimension() == 4:
            tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2] = q
        else:
            tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2, (tensor.size(4)-1)//2] = q
        tensor.mul_(math.sqrt(gain))
    return tensor