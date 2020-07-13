import torch
import copy
import numpy
from torch import cuda, nn
from utils.cutmix import rand_bbox

__all__ = ['mixup', 'cutmix']

def mixup(images, labels, device, alpha=1.0):
    """
    mixup function from 'mixup: BEYOND EMPIRICAL RISK MINIMIZATION', 
    https://arxiv.org/pdf/1710.09412.pdf
    """
        
    lam = numpy.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size()[0]).to(device)
    labels1 = labels
    labels2 = labels[rand_index]
    images2 = copy.deepcopy(images)
            
    images = Variable(lam * images + (1-lam)*images2[rand_index,:,:,:]).to(device)
    
    return lam, images, labels1, labels2

def cutmix(images, labels, device, alpha = 1.0):
    """
    cutmix function from 'CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features',
    https://arxiv.org/abs/1905.04899
    """
    
    #generate mixed sample
    lam = numpy.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size()[0]).to(device)
    labels_a = labels
    labels_b = labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    #adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    #compute output
    images = torch.autograd.Variable(images, requires_grad=True).to(device)

    return lam, images, labels_a, labels_b

    
