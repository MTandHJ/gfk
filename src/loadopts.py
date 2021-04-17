

from typing import TypeVar, Callable, Optional, Tuple, Dict, cast, NoReturn
import torch
import torchvision
import torchvision.transforms as T

import numpy as np
import os

from functools import partial
from tqdm import tqdm



from .dict2obj import Config
from .config import *



class ModelNotDefineError(Exception): pass
class LossNotDefineError(Exception): pass
class OptimNotIncludeError(Exception): pass
class DatasetNotIncludeError(Exception): pass


# return the num_classes of corresponding dataset
def get_num_classes(dataset_type: str) -> int:
    try:
        return NUMCLASSES[dataset_type]
    except KeyError:
        raise DatasetNotIncludeError("Dataset {0} is not included." \
                        "Refer to the following: {1}".format(dataset_type, _dataset.__doc__))


# return the shape of corresponding dataset
def get_shape(dataset_type: str) -> Tuple:
    try:
        return SHAPES[dataset_type]
    except:
        raise DatasetNotIncludeError("Dataset {0} is not included." \
                        "Refer to the following: {1}".format(dataset_type, _dataset.__doc__))


def load_model(model_type: str) -> Callable:
    types = {
        "g": "Gen",
        "d": "Dis"
    }
    name, model_type = model_type.split("-")
    module_name = "models." + name
    try:
        model= types[model_type]
        module = __import__(module_name)
        model = getattr(getattr(module, name), model)
    except AttributeError:
       raise ModelNotDefineError(f"model {model_type} is not defined.\n" \
                    f"Refer to the following: {load_model.__doc__}\n")
    return cast(Callable, model)


def refine_model(
    model: torch.nn.Module, init_policy: Optional[str] = None, 
    need_sn: bool = False, name: str = "weight",
    n_power_iterations: int = 1, eps: float = 1e-12,
    dim: Optional[int] = None,
    basic_keys: Optional[set] = None,
    except_keys: Optional[set] = None,
) -> NoReturn:
    from models.utils import spectral_norm, init_weights
    if need_sn:
        spectral_norm(
            model, name,
            n_power_iterations=n_power_iterations,
            eps=eps, dim=dim,
            basic_keys=basic_keys, except_keys=except_keys
        )
    init_weights(model, init_policy=init_policy)


def load_inception_model(
    resize: bool = True
):
    from metrics.utils import load_inception
    return load_inception(resize=resize)[0]


def load_loss_func(
    loss_type: str, 
    mode: str,
    reduction="mean",
    **kwargs
) -> Callable[..., torch.Tensor]:
    """
    loss_type:
        bce: binary cross entropy
    mode:
        gen or dis
    reduction:
        mean or sum
    """
    loss_func: Callable
    if loss_type == "bce":
        from .loss_zoo import BCELoss
        loss_func = BCELoss(mode=mode, reduction=reduction)
    elif loss_type == "hinge":
        from .loss_zoo import HingeLoss
        loss_func = HingeLoss(mode=mode, reduction=reduction)
    elif loss_type == "ls":
        from .loss_zoo import LeastSquaresLoss
        loss_func = LeastSquaresLoss(mode=mode, reduction=reduction)
    elif loss_type == "wloss":
        from .loss_zoo import WLoss
        loss_func = WLoss(mode=mode, reduction=reduction)
    else:
        raise LossNotDefineError(f"Loss {loss_type} is not defined.\n" \
                    f"Refer to the following: {load_loss_func.__doc__}")
    return loss_func


def _get_transform(
    dataset_type: str, 
    transform: str, 
) -> "transform":
    return TRANSFORMS[dataset_type]


def _dataset(
    dataset_type: str, 
    transform: str,  
    train: bool = True
) -> torch.utils.data.Dataset:
    """
    Dataset:
    mnist: MNIST
    cifar10: CIFAR-10
    cifar100: CIFAR-100
    celeba: CelebA
    Transform:
    default: the default transform for each data set
    """
    try:
        transform = _get_transform(dataset_type, transform)
    except KeyError:
        raise DatasetNotIncludeError(f"Dataset {dataset_type} or transform {transform} is not included.\n" \
                        f"Refer to the following: {_dataset.__doc__}")

    if dataset_type == "mnist":
        dataset = torchvision.datasets.MNIST(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    elif dataset_type == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    elif dataset_type == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    elif dataset_type == "celeba":
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(ROOT, dataset_type),
            transform=transform
        )
        
    return dataset

def load_augmentor(aug_policy: str = 'null', channels_first: bool = True):
    """
    "null": no augmentation
    "diff_aug": differentiable augmentation
    """
    print(f">>> Applying augmentations: {aug_policy} ...")
    try:
        augmentor = AUGMENTATIONS[aug_policy]

    except KeyError:
        raise NotImplementedError(f"No such augmentation: {aug_policy} ...\n" \
            f"Refer to the following: {load_augmentor.__doc__}")
    finally:
        return augmentor


def load_dataset(
    dataset_type: str, 
    transform: str ='default', 
    train: bool = True
) -> torch.utils.data.Dataset:
    dataset = _dataset(dataset_type, transform, train)
    return dataset


class _TQDMDataLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        return iter(
            tqdm(
                super(_TQDMDataLoader, self).__iter__(), 
                leave=False, desc="վ'ᴗ' ի-"
            )
        )


def load_dataloader(
    dataset: torch.utils.data.Dataset, 
    batch_size: int, 
    train: bool = True, 
    show_progress: bool = False
) -> torch.utils.data.DataLoader:

    dataloader = _TQDMDataLoader if show_progress else torch.utils.data.DataLoader
    if train:
        dataloader = dataloader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY)
    else:
        dataloader = dataloader(dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY)

    return dataloader


def load_optimizer(
    model: torch.nn.Module, 
    optim_type: str, *,
    lr: float = 0.1, momentum: float = 0.9,
    betas: Tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 1e-4,
    nesterov: bool = False,
    **kwargs: "other hyper-parameters for optimizer"
) -> torch.optim.Optimizer:
    """
    sgd: SGD
    adam: Adam
    """
    try:
        cfg = OPTIMS[optim_type]
    except KeyError:
        raise OptimNotIncludeError(f"Optim {optim_type} is not included.\n" \
                        f"Refer to the following: {load_optimizer.__doc__}")
    
    kwargs.update(lr=lr, momentum=momentum, betas=betas, 
                weight_decay=weight_decay, nesterov=nesterov)
    
    cfg.update(**kwargs) # update the kwargs needed automatically
    print(optim_type, cfg)
    if optim_type == "sgd":
        optim = torch.optim.SGD(model.parameters(), **cfg)
    elif optim_type == "adam":
        optim = torch.optim.Adam(model.parameters(), **cfg)

    return optim


def load_learning_policy(
    optimizer: torch.optim.Optimizer,
    learning_policy_type: str,
    **kwargs: "other hyper-parameters for learning scheduler"
) -> "learning policy":
    """
    default: (100, 105), 110 epochs suggested
    null:
    cosine: CosineAnnealingLR, kwargs: T_max, eta_min, last_epoch
    """
    try:
        learning_policy_ = LEARNING_POLICY[learning_policy_type]
    except KeyError:
        raise NotImplementedError(f"Learning_policy {learning_policy_type} is not defined.\n" \
            f"Refer to the following: {load_learning_policy.__doc__}")

    lp_type = learning_policy_[0]
    lp_cfg = learning_policy_[1]
    lp_description = learning_policy_[2]
    lp_cfg.update(**kwargs) # update the kwargs needed automatically
    print(lp_type, lp_cfg)
    print(lp_description)
    learning_policy = getattr(
        torch.optim.lr_scheduler, 
        lp_type
    )(optimizer, **lp_cfg)
    
    return learning_policy


def load_sampler(
    rtype: str, *,
    low: float = 0.,
    high: float = 1.,
    loc: float = 0.,
    scale: float = 1.,
    threshold: float = 1.
):
    """
    uniform: (low, high)
    normal: (loc, scale)
    truncate normal: threshold
    """
    if rtype == "uniform":
        sampler = torch.distributions.uniform.Uniform(low=low, high=high).sample
    elif rtype == "normal":
        sampler = torch.distributions.normal.Normal(loc=loc, scale=scale).sample
    elif rtype == "tnormal":
        from scipy.stats import truncnorm
        def truncated_normal(size, threshold=1.):
            values = truncnorm.rvs(-threshold, threshold, size=size)
            return torch.FloatTensor(values)
        sampler = partial(truncated_normal, threshold=threshold)
    else:
        raise NotImplementedError(f"No such sampler: {rtype}. \n " \
            f"Refer to the following: {load_sampler.__doc__}")
    return sampler


def generate_path(
    method: str, dataset_type: str, 
    generator: str, discriminator: str,  
    description: str
) -> Tuple[str, str]:
    info_path = INFO_PATH.format(
        method=method,
        dataset=dataset_type,
        generator=generator,
        discriminator=discriminator,
        description=description
    )
    log_path = LOG_PATH.format(
        method=method,
        dataset=dataset_type,
        generator=generator,
        discriminator=discriminator,
        description=description
    )
    return info_path, log_path

