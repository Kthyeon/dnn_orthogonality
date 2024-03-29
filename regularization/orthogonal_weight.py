import torch
from torch.optim import lr_scheduler
import copy
from torch import cuda, nn, optim
from tqdm import tqdm, trange
import numpy
from torch.nn.functional import normalize
from torch.autograd import Variable
from torch import cuda, nn, optim
import torch.nn.functional as F

__all__ = ['wd_reg', 'norm_reg', 'psrip_reg','or_reg','ortho_reg', 'ortho']

def conv_ortho(weight, device):    
    cols = weight[0].numel()
    w1 = weight.view(-1, cols)
    wt = torch.transpose(w1, 0, 1)
    m = torch.matmul(wt, w1)
    ident = Variable(torch.eye(cols, cols)).to(device)

    w_tmp = (m-ident)
    sigma = torch.norm(w_tmp)
    
    return sigma


#####################################################################################################
#####################################################################################################

# def depth_ortho(weight, tp = 'app'):
#     # tp: app (근사) & ori (있는 그대로)
#     l2_reg = 0
#     if tp =='app':
#         for W in weight:
#             tmp = 0
#             W = W.squeeze()
#             for row in W:
#                 tmp += torch.sum(torch.square(row))
#             l2_reg += torch.abs(tmp-1)
#     elif tp=='ori':
#         for W in weight:
#             tmp = 0
#             W = W.squeeze()
#             for row in W:
#                 tmp += torch.sum(torch.square(row))
#             l2_reg += torch.abs(tmp-1)
#             l2_reg += torch.abs(W[0,0]*W[0,1]+W[0,1]*W[0,2]+W[1,0]*W[1,1]+W[1,1]*W[1,2]+W[2,0]*W[2,1]+W[2,1]*W[2,2])
#             l2_reg += torch.abs(W[0,0]*W[0,2]+W[1,0]*W[1,2]+W[2,0]*W[2,2])
#     return l2_reg

#####################################################################################################
#####################################################################################################

def wd_reg(mdl, device, lamb_list=[0.0, 1.0, 0.0, 0.0], opt='both'):
    
    assert type(lamb_list) is list, 'lamb_list should be list'
    l2_reg = None
    
    for module in mdl.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv2d):
                if module.weight.shape[2] == 1:
                    if opt == 'exp' and module.weight.shape[0]<module.weight.shape[1]:
                        continue
                    elif opt == 'rec' and module.weight.shape[0]>module.weight.shape[1]:
                        continue
                    # pointwise conv
                    W = module.weight
                    lamb = lamb_list[1]
                elif module.weight.shape[2] ==3:
                    W = module.weight
                    if module.weight.shape[1] == 1:
                        # Depthwise convolution.
                        lamb = lamb_list[2]
                    else:
                        # Original convolution. (Maybe including stem conv)
                        lamb = lamb_list[0]
            elif isinstance(module, nn.Linear):
                # fully connected layer
                W = module.weight
                lamb = lamb_list[3]
            else:
                continue
            
            if l2_reg is None:
                l2_reg = lamb * torch.sum(W.square()) / 2
            else:
                l2_reg += lamb * torch.sum(W.square()) / 2
        else:
            continue
    
    return l2_reg

#####################################################################################################
#####################################################################################################

# This is for novel method about rigorously orthogonalizing the weight parameteres in conv layer.
# But, it is in the undeveloped stage, and most parts are under complete.
def gram_schmidt(vectors,device):
    #to make pointwise matrix independent matrix
    if vectors.shape[0]>vectors.shape[1]:
        vectors = vectors.transpose(0,1)
    basis = torch.zeros_like(vectors).to(device)
    for num in range(vectors.shape[0]):
        temp = torch.zeros_like(vectors[num])
        for b in basis:
            temp += torch.matmul(vectors[num],b) * b
        w = vectors[num] - temp
        if (w > 1e-10).any():  
            basis[num] = w/torch.norm(w)
    basis = basis.half()
    
    if vectors.shape[0]>vectors.shape[1]:
        return basis.transpose(0,1) 
    else:
        return basis
    
def gr_sch_pr(mdl, device):
    for name, module in mdl.named_children():
        if 'layer' in name:
            for m in module:
                m.conv1.weight = torch.nn.Parameter(gram_schmidt(m.conv1.weight, device))
                m.conv3.weight = torch.nn.Parameter(gram_schmidt(m.conv3.weight, device))

#####################################################################################################   
#####################################################################################################

# This is a traditional method of regularization which penalizes the semi-orthogonality of weight matrices.
# This concept is well-known and widely-used method.

def norm_reg(mdl, device, lamb_list=[0.0, 1.0, 0.0, 0.0], opt = 'both'):
    # Consider the below facotrs.
    # factor1: which kind layer (e.g., pointwise, depthwise, original, fc_layer)
    # factor2: power of regularization (i.e., lambda). Maybe, we should differ from each class of layer's lambda.
    # How do we handle these?
    # 'lamb_list' is a list of hyperparmeters for each class of layer. [origin_conv, pointwise, depthwise, fully coneected layer]
    # 'lamb_list' length is 4.
    # 'opt' : position of pointwise convolution. - expansion stage (exp), reduction stage (rec), both.

    assert type(lamb_list) is list, 'lamb_list should be list.'
    l2_reg = None

    for module in mdl.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv2d):
                if module.weight.shape[2] == 1:
                    if opt == 'exp' and module.weight.shape[0]>module.weight.shape[1]:
                        continue
                    elif opt == 'rec' and module.weight.shape[0]<module.weight.shape[1]:
                        continue
                    # pointwise conv
                    W = module.weight
                    lamb = lamb_list[1]
                elif module.weight.shape[2] ==3:
                    W = module.weight
                    if module.weight.shape[1] == 1:
                        # Depthwise convolution.
                        lamb = lamb_list[2]
                        W = module.weight
#                         if l2_reg is None:
#                             l2_reg = lamb * conv_ortho(W)
#                             num = 1
#                         else:
#                             l2_reg += lamb * conv_ortho(W)
#                             num += 1
#                         continue
                    else:
                        # Original convolution. (Maybe including stem conv)
                        lamb = lamb_list[0]
                        W = module.weight
#                         if l2_reg is None:
#                             l2_reg = lamb * conv_ortho(W, device)
#                             num = 1
#                         else:
#                             l2_reg += lamb * conv_ortho(W, device)
#                             num += 1
#                         continue
            elif isinstance(module, nn.Linear):
                # fully connected layer
                W = module.weight
                lamb = lamb_list[3]
            else:
                continue

            cols = W[0].numel()
            w1 = W.view(-1, cols) # W.shape[1] * W.shape[0]
            wt = torch.transpose(w1, 0, 1)
            if W.shape[0]< W.shape[1]:
                m = torch.matmul(wt, w1)
                ident = Variable(torch.eye(cols, cols)).to(device)
            else:
                m = torch.matmul(w1, wt)
                ident = Variable(torch.eye(m.shape[0], m.shape[0])).to(device)

            w_tmp = (m-ident)
            sigma = torch.norm(w_tmp)
            
            if l2_reg is None:
                l2_reg = lamb * sigma**2
                num = 1
            else:
                l2_reg += lamb * sigma**2
                num += 1
        else:
            continue
    return l2_reg / num

#####################################################################################################
#####################################################################################################   
# This is from the work: 'Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?,' 
# https://arxiv.org/abs/1810.09102.

def psrip_reg(mdl, device, lamb_list=[0.0, 1.0, 0.0, 0.0], opt='both', tp='app', level='iden'):
    # Consider the below facotrs.
    # factor1: which kind layer (e.g., original(stem), pointwise, depthwise, fc_layer)
    # factor2: power of regularization (i.e., lambda). Maybe, we should differ from each class of layer's lambda.
    # How do we handle these?
    # 'lamb_list' is a list of hyperparmeters for each class of layer. [origin_conv, pointwise, depthwise, fully coneected layer]
    # 'lamb_list' length is 4.
    # 'opt' : position of pointwise convolution. - expansion stage (exp), reduction stage (rec), both.
    
    assert type(lamb_list) is list and len(lamb_list)==4, 'lamb_list should be list.'
    l2_reg = None

    for module in mdl.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv2d):
                if module.weight.shape[2] == 1:
                    if opt == 'exp' and module.weight.shape[0]>module.weight.shape[1]:
                        continue
                    elif opt == 'rec' and module.weight.shape[0]<module.weight.shape[1]:
                        continue
                    # pointwise conv
                    W = module.weight
                    lamb = lamb_list[1]
                elif module.weight.shape[2] ==3:
                    W = module.weight
                    if module.weight.shape[1] == 1:
                        # Depthwise convolution.
                        lamb = lamb_list[2]
                        W = module.weight
#                         if l2_reg is None:
#                             l2_reg = lamb * conv_ortho(W, tp)
#                             num = 1
#                         else:
#                             l2_reg += lamb * conv_ortho(W, tp)
#                             num += 1
#                         continue
                    else:
                        # Original convolution. (Maybe including stem conv)
                        lamb = lamb_list[0]
                        W = module.weight
#                         if l2_reg is None:
#                             l2_reg = lamb * conv_ortho(W, device)
#                             num = 1
#                         else:
#                             l2_reg += lamb * conv_ortho(W, device)
#                             num += 1
#                         continue
            elif isinstance(module, nn.Linear):
                # fully connected layer
                W = module.weight
                lamb = lamb_list[3]
            else:
                continue
                
            cols = W[0].numel()
            w1 = W.view(-1, cols) # W.shape[1] * W.shape[0]
            wt = torch.transpose(w1, 0, 1)
            if W.shape[0]< W.shape[1]:
                m = torch.matmul(wt, w1)
                ident = Variable(torch.eye(cols, cols)).to(device)
            else:
                m = torch.matmul(w1, wt)
                ident = Variable(torch.eye(m.shape[0], m.shape[0])).to(device)
   
            
            if level == 'iden':
                w_tmp = (m-ident)
            else:
                w_tmp = w1
            height = w_tmp.size(0)
            u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
            v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
            u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
            sigma = torch.dot(u, torch.matmul(w_tmp, v))
            
            if l2_reg is None:
                l2_reg = lamb * (sigma)**2
                num = 1
            else:
                l2_reg += lamb * (sigma)**2
                num += 1
        else:
            continue
    

    return l2_reg/num



def ortho(mdl, device):
    ortho_penalty = []
    cnt = 0
    for m in mdl.modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (7, 7) or m.weight.shape[1] == 3:
                continue
            o = ortho_conv(m, device)
            cnt += 1
            ortho_penalty.append(o)
    ortho_penalty = sum(ortho_penalty)
    return ortho_penalty

def ortho_conv(m, device):
    operator = m.weight
    operand = torch.cat(torch.chunk(m.weight, m.groups, dim=0), dim=1)
    transposed = m.weight.shape[1] < m.weight.shape[0]
    num_channels = m.weight.shape[1] if transposed else m.weight.shape[0]
    if transposed:
        operand = operand.transpose(1, 0)
        operator = operator.transpose(1, 0)
    gram = F.conv2d(operand, operator, padding=(m.kernel_size[0] - 1, m.kernel_size[1] - 1),
                    stride=m.stride, groups=m.groups)
    identity = torch.zeros(gram.shape).to(device)
    identity[:, :, identity.shape[2] // 2, identity.shape[3] // 2] = torch.eye(num_channels).repeat(1, m.groups)
    out = torch.sum((gram - identity) ** 2.0) / 2.0
    return out


#####################################################################################################
#####################################################################################################
# This is a code for orthogonalizing the weight.
# Here is no characteristic of normalization. To confirm whether ORN has norm-preservation property.
# Implement the reg with SRIP

def or_reg(mdl, device, lamb_list=[0.0, 1.0, 0.0, 0.0], opt = 'both'):
    # Consider the below facotrs.
    # factor1: which kind layer (e.g., pointwise, depthwise, original, fc_layer)
    # factor2: power of regularization (i.e., lambda). Maybe, we should differ from each class of layer's lambda.
    # How do we handle these?
    # 'lamb_list' is a list of hyperparmeters for each class of layer. [origin_conv, pointwise, depthwise, fully coneected layer]
    # 'lamb_list' length is 4.
    # 'opt' : position of pointwise convolution. - expansion stage (exp), reduction stage (rec), both.
    
    assert type(lamb_list) is list, 'lamb_list should be list.'
    l2_reg = None

    for module in mdl.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv2d):
                if module.weight.shape[2] == 1:
                    if opt == 'exp' and module.weight.shape[0]<module.weight.shape[1]:
                        continue
                    elif opt == 'rec' and module.weight.shape[0]>module.weight.shape[1]:
                        continue
                    # pointwise conv
                    W = module.weight
                    lamb = lamb_list[1]
                elif module.weight.shape[2] ==3:
                    W = module.weight
                    if module.weight.shape[1] == 1:
                        # Depthwise convolution.
                        lamb = lamb_list[2]
                    else:
                        # Original convolution. (Maybe including stem conv)
                        lamb = lamb_list[0]
            elif isinstance(module, nn.Linear):
                # fully connected layer
                W = module.weight
                lamb = lamb_list[3]
            else:
                continue

            cols = W[0].numel() 
            w1 = W.view(-1, cols) # out_channels x all
            wt = torch.transpose(w1, 0, 1)
            m = torch.matmul(wt, w1)
            
#             w_tmp = (m - torch.diagflat(torch.diagonal(m)))
#             height = w_tmp.size(0)
#             u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
#             v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
#             u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
#             sigma = torch.dot(u, torch.matmul(w_tmp, v))
            row_sigma = torch.norm(m, p=1, dim=1)
            w_tmp = row_sigma - torch.ones_like(row_sigma)
            
            sigma = torch.norm(w_tmp, p=1)
            
            if l2_reg is None:
                l2_reg = lamb * (sigma)
                num = 1
            else:
                l2_reg += lamb * (sigma)
                num += 1
        else:
            continue

    return l2_reg / num

#####################################################################################################
#####################################################################################################

# What's next?
# How to apply the regularization with noise?
# How about regularizing the weight matrix perturbed by gaussian random noise?

def ortho_reg(mdl, device, lamb_list=[0.0, 1.0, 0.0, 0.0], opt = 'both'):
    # Consider the below facotrs.
    # factor1: which kind layer (e.g., pointwise, depthwise, original, fc_layer)
    # factor2: power of regularization (i.e., lambda). Maybe, we should differ from each class of layer's lambda.
    # How do we handle these?
    # 'lamb_list' is a list of hyperparmeters for each class of layer. [origin_conv, pointwise, depthwise, fully coneected layer]
    # 'lamb_list' length is 4.
    # 'opt' : position of pointwise convolution. - expansion stage (exp), reduction stage (rec), both.
    
    assert type(lamb_list) is list, 'lamb_list should be list.'
    l2_reg = None

    for module in mdl.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv2d):
                if module.weight.shape[2] == 1:
                    if opt == 'exp' and module.weight.shape[0]<module.weight.shape[1]:
                        continue
                    elif opt == 'rec' and module.weight.shape[0]>module.weight.shape[1]:
                        continue
                    # pointwise conv
                    W = module.weight
                    lamb = lamb_list[1]
                elif module.weight.shape[2] ==3:
                    W = module.weight
                    if module.weight.shape[1] == 1:
                        # Depthwise convolution.
                        lamb = lamb_list[2]
                    else:
                        # Original convolution. (Maybe including stem conv)
                        lamb = lamb_list[0]
            elif isinstance(module, nn.Linear):
                # fully connected layer
                W = module.weight
                lamb = lamb_list[3]
            else:
                continue

            cols = W[0].numel() 
            w1 = W.view(-1, cols) # out_channels x all
            wt = torch.transpose(w1, 0, 1)
            m = torch.matmul(wt, w1)

            w_tmp = (m-torch.diag(m))
#             height = w_tmp.size(0)
#             u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
#             v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
#             u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
#             sigma = torch.dot(u, torch.matmul(w_tmp, v))
            sigma = torch.norm(w_tmp)
            if l2_reg is None:
                l2_reg = lamb * (sigma)**2
                num = 1
            else:
                l2_reg += lamb * (sigma)**2
                num += 1
        else:
            continue

    return l2_reg / num


#####################################################################################################
#####################################################################################################
