import torch
import copy
from torch import cuda, nn

class LabelSmoothingLoss(nn.Module):
    # Label smoothing method: https://arxiv.org/abs/1512.00567
    # It injects the uniform noise to the hard target (i.e., one-hot vector) whose magnitude is epsilon.
    def __init__(self, device, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.device = device

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred).to(self.device)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
