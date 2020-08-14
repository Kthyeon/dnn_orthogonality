'''EfficientNetB0

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
from efficientnet_pytorch import EfficientNet

class EfficientNetB0(nn.Module):
    def __init__(self, opt='both', init = 'xavier', pre_trained=False, num_classes =1000):
        # opt: option that orthogonalizes the pointwise conv in either expension or reduction or both.
        # opt in ['exp', 'rec', 'both']
        # init in ['xavier', 'kaiming', 'ort', 'z_ort']
        super(EfficientNetB0, self).__init__()
        self.init = init
        self.opt = opt
        if pre_trained:
            self.net = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.net = EfficientNet.from_name('efficientnet-b0')
        self.net._fc = nn.Linear(1280, num_classes, bias = True)

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
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.weight.shape[2] == 1:
                        nn.init.orthogonal_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
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
    
    def extract_feature(self, x):
        # To analyze the behavior of layerwise response.
        # feature_list: layerwise output whose channel size is 1. (kind of activatoin map) (shape: NxNx1)
        # map_list: avgpool of each instance (shape: 1x1xN)
        # norm_list: norm of each instance (2-norm)
        feature_list = []
        map_list = []
        norm_list = []

        out = self.net._bn0(self.net._conv_stem(x))
        for block in self.net._blocks:
            tmp_f = []
            tmp_m = []
            tmp_n = []
            # input
            tmp_f.append(F.adaptive_avg_pool2d(F.adaptive_avg_pool2d(out,2), 3))
            tmp_m.append(F.adaptive_avg_pool2d(out, 1))
            tmp_n.append(torch.norm(out,2))

            out1 = block._swish(block._bn0(block._expand_conv(out)))
            # after expand conv
            tmp_f.append(F.adaptive_avg_pool2d(F.adaptive_avg_pool2d(out1,2), 3))
            tmp_m.append(F.adaptive_avg_pool2d(out1, 1))
            tmp_n.append(torch.norm(out1,2))

            out1 = block._swish(block._bn1(block._depthwise_conv(out1)))
            if block._block_args.expand_ratio != 1:
                out_s = F.adaptive_avg_pool2d(out1, 1)
                out_s = block._se_reduce(out_s)
                out_s = blcok._swish(out_s)
                out_s = block._se_expand(out_s)
                out1 = torch.sigmoid(out_s) * out1
            out1 = block._bn2(block._project_conv(out1))
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

    

