


from typing import Tuple, List, Optional, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3, Inception3

import numpy as np
from  sklearn.model_selection import KFold

import os
from functools import partial

from .config import *
from src.utils import load, gpu


def _feature_hook(module, input, output, log):
    log.features = input[0]

class Net(nn.Module):

    def __init__(
        self, arch: nn.Module, 
        resize: bool = True, normalizer: Callable = None
    ):
        super(Net, self).__init__()

        self.features = None
        self.arch = arch
        self.arch.fc.register_forward_hook(
            partial(_feature_hook, log=self)
        )
        self.resize = resize
        if normalizer is None:
            self.normalizer = lambda x: x
        else:
            self.normalizer = normalizer

    def _preprocessing(self, inputs):
        n, c, h, w = inputs.size()
        if c is 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        inputs = self.normalizer(inputs)
        if self.resize:
            inputs = F.interpolate(
                inputs,
                size=(299, 299),
                mode='bilinear',
                align_corners=False
            )
        return inputs
 

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        inputs = self._preprocessing(inputs)
        logits = self.arch(inputs)
        return self.features, logits


def load_inception(
    resize: bool = True, normalizer: Optional[Callable] = None
) -> Tuple[nn.Module, torch.device]:

    files = os.listdir(path=PATH)
    filename = None
    for file_ in files:
        if file_.startswith(INCEPTION_V3):
            filename = file_
            break
    if filename is None:
        print(">>> No pre_trained model is found. Download from url ...")
        inception_model = inception_v3(pretrained=True, transform_input=False)
        device, gpu(inception_model)
    else:
        print(f">>> Pre_trained model is found: {filename} ...")
        inception_model = Inception3(transform_input=False)
        device = gpu(inception_model)
        load(
            model=inception_model,
            path=PATH,
            filename=filename,
            device=device
        )
    model = Net(arch=inception_model, resize=resize, normalizer=normalizer)
    return model, device
        

def kldiv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    p: N x d
    q: N x d
    return : N
    """
    p_log = np.log(p + 1e-12)
    q_log = np.log(q + 1e-12)
    div = p * (p_log - q_log)
    return div.sum(axis=1)
