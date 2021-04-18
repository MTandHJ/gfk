


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


def _feature_hook(module, input, output, logger):
    logger.features = input[0]

class _Net(nn.Module):

    def __init__(
        self, arch: nn.Module, 
        resize: bool = True, normalizer: Callable = None
    ):
        super(_Net, self).__init__()

        self.features = None
        self.arch = arch
        self.arch.fc.register_forward_hook(
            partial(_feature_hook, logger=self)
        )
        self.resize = resize
        if normalizer is None:
            self.normalizer = nn.Identity()
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

    file_ = os.path.join(INFO_PATH, INCEPTION_V3)
    if not os.path.exists(file_):
        print(">>> Inception model is not found. Download from url ...")
        inception_model = inception_v3(pretrained=True, transform_input=False)
        device = gpu(inception_model)
        torch.save(
            inception_model.state_dict(),
            file_
        )
    else:
        print(f">>> Inception model is found: {file_} ...")
        inception_model = Inception3(transform_input=False)
        device = gpu(inception_model)
        load(
            model=inception_model,
            path=INFO_PATH,
            filename=INCEPTION_V3,
            device=device
        )
    model = _Net(arch=inception_model, resize=resize, normalizer=normalizer)
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
