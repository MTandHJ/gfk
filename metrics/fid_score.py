

"""
Reference:
mseitzer: pytorch-fid
https://github.com/mseitzer/pytorch-fid
"""

from typing import Tuple, Iterable, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import linalg

import os

from .config import INFO_PATH
from .utils import load_inception
from src.loadopts import load_dataset, load_dataloader
from src.utils import import_pickle, export_pickle


POSTFIX = "fid"

@torch.no_grad()
def _step(dataloader: Iterable, model: nn.Module, device: torch.device) -> Tuple[np.ndarray]:
    model.eval()
    preds = []
    for imgs in dataloader:
        imgs = imgs.to(device)
        features, _ = model(imgs)
        preds.append(features.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    np.random.shuffle(preds) # shuffle

    mu = np.mean(preds, axis=0)
    cov = np.cov(preds, rowvar=False) # per column is a variable
    return mu, cov


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean).item()


def fid_score_single(
    dataloader: Iterable,
    dataset_type: str,
    model: nn.Module,
    device: torch.device,
) -> Tuple[float]:
    """
    'single' implies that we calculate mean and cov for the first dataset,
    loading the mean and cov of the second one if could.
    dataloader: IgnoreLabelDataset
    dataset_type: such "cifar10", 
    model: the incpetion v3 model
    device:
    """

    try:
        infos = import_pickle(os.path.join(INFO_PATH, ".".join((dataset_type, POSTFIX))))
    except ImportError:
        infos = prepare_fids(dataset_type, model, device)
    mu1, cov1 = infos['mu'], infos['cov']
    mu2, cov2 = _step(dataloader, model, device)
    return calculate_frechet_distance(mu1, cov1, mu2, cov2)


 
def fid_score_double(
    dataloader1: Iterable,
    dataloader2: Iterable,
    model: nn.Module,
    device: torch.device,
) -> Tuple[float]:

    mu1, cov1 = _step(dataloader1, model, device)
    mu2, cov2 = _step(dataloader2, model, device)

    return calculate_frechet_distance(mu1, cov1, mu2, cov2)


def prepare_fids(
    dataset_type: str, inception_model: nn.Module, 
    device: torch.device, batch_size: int = 16
) -> None:
    """
    Prepare mean and cov of real dataset.
    The saved filename ends with ".fid".
    """
    from .datasets import IgnoreLabelDataset
    print(f">>> Prepare FID score of {dataset_type} ...")
    dataset = load_dataset(dataset_type)
    dataset = IgnoreLabelDataset(dataset)
    dataloader = load_dataloader(dataset, batch_size=batch_size)

    mu, cov = _step(dataloader, inception_model, device)
    infos = {'mu': mu, 'cov': cov}
    filename = os.path.join(INFO_PATH, ".".join((dataset_type, POSTFIX)))
    export_pickle(data=infos, filename=filename)
    return infos

    
