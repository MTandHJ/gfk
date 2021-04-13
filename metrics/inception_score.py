

"""
Reference:
sbarratt: inception-score-pytorch
https://github.com/sbarratt/inception-score-pytorch
"""

from typing import Tuple, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .utils import load_inception, kldiv


@torch.no_grad()
def inception_score(
    dataloader: Iterable, 
    model: nn.Module,
    device: torch.device,
    n_splits: int = 1,
):
    """
    dataloader: IgnoreLabelDataset
    model: inception v3 model
    device:
    n_splits: splits the dataset into n_splits and
    return: the mean and std
    """

    model.eval()
    preds = []
    for imgs in dataloader:
        imgs = imgs.to(device)
        _, logits = model(imgs)
        preds.append(F.softmax(logits, dim=1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    np.random.shuffle(preds) # shuffle

    scores = []
    step = preds.shape[0] // n_splits
    for k in range(n_splits):
        start, end = k * step, min((k + 1) * step, preds.shape[0])
        pyx = preds[start:end]
        py = pyx.mean(axis=0)
        scores.append(np.exp(kldiv(pyx, py).mean()))
    
    return np.mean(scores).item(), np.std(scores).item()


        





