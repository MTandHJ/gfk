


# Here are some basic settings.
# It could be overwritten if you want to specify
# special configs. However, please check the correspoding
# codes in loadopts.py.



import torch.nn as nn
import torchvision.transforms as T

import os

from .augmentation import DiffAug
from .dict2obj import Config



# for saving
ROOT = "../data"
HDF5 = os.path.join(ROOT, "hdf5data")
HDF5_POSFIX = "hdf5"
INFO_PATH = "./infos/{method}/{dataset}-{generator}-{discriminator}/{description}"
LOG_PATH = "./logs/{method}/{dataset}-{generator}-{discriminator}/{description}-{time}"
TIMEFMT = "%m%d%H"
SAVED_FILENAME = "Generator_paras.pt" # the filename of saved model



NUMCLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100
}

SHAPES = {
    "mnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "celeba": (3, 64, 64)
}


TRANSFORMS = {
    "mnist": [T.ToTensor()],
    "cifar10": [T.ToTensor()],
    "cifar100": [T.ToTensor()],
    "celeba": [ 
        T.Resize(SHAPES['celeba'][1:]),
        T.CenterCrop(SHAPES['celeba'][1:]),
        T.ToTensor()
    ]
}

NORMALIZATIONS = {
    "mnist": T.Normalize((0.5,), (0.5,)),
    "cifar10": T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "cifar100": T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "celeba": T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}

AUGMENTATIONS = {
    "null": nn.Identity(),
    "diff_aug": DiffAug(policy="color,translation,cutout")
}

# env settings
NUM_WORKERS = 3
PIN_MEMORY = True

# the settings of optimizers of which lr could be pointed
# additionally.
OPTIMS = {
    "sgd": Config(lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=False, prefix="SGD:"),
    "adam": Config(lr=0.01, betas=(0.9, 0.999), weight_decay=0., prefix="Adam:")
}


# the learning schedular can be added here
LEARNING_POLICY = {
    "null": (
        "StepLR",
        Config(
            step_size=9999999999999,
            gamma=1,
            prefix="Null leaning policy will be applied:\n"
        )
    ),
   "cosine":(   
        "CosineAnnealingLR",   
        Config(          
            T_max=100000,
            eta_min=0.,
            last_epoch=-1,
            prefix="cosine learning policy: T_max == epochs - 1"
        )
    )
}








