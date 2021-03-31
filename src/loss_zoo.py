

import torch
import torch.nn as nn
import torch.nn.functional as F



def cross_entropy(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    reduction: str = "mean"
) -> torch.Tensor:
    return F.cross_entropy(logits, targets, reduction=reduction)

def bce_loss(
    probs: torch.Tensor, 
    targets: torch.Tensor, 
    reduction: str = "mean"
) -> torch.Tensor:
    return F.binary_cross_entropy(probs, targets, reduction=reduction)

def mse_loss(
    x: torch.Tensor, y: torch.Tensor, 
    reduction: str = "mean"
) -> torch.Tensor:
    return F.mse_loss(x, y, reduction=reduction)

class Energy(nn.Module):
    
    def __init__(self, margin: float = 10.):
        super(Energy, self).__init__()
        self.margin = margin

    def forward(
        self, outs: torch.Tensor, 
        labels: torch.Tensor, 
        reduction: str = "mean"
    ):
        m = self.margin
        part1 = outs * labels
        part2 = F.relu(m - outs) * (1 - labels)
        loss = part1 + part2
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()