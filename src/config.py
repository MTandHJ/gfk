


# Here are some basic settings.
# It could be overwritten if you want to specify
# special configs. However, please check the correspoding
# codes in loadopts.py.



import torch.nn as nn
import torchvision.transforms as T

from .augmentation import DiffAug
from .dict2obj import Config



# for saving
ROOT = "../data"
INFO_PATH = "./infos/{method}/{dataset}-{generator}-{discriminator}/{description}"
LOG_PATH = "./logs/{method}/{dataset}-{generator}-{discriminator}/{description}"
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
    "celeba": (3, 218, 178)
}


TRANSFORMS = {
    "mnist": T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "cifar10": T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "cifar100": T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "celeba": T.Compose([
        T.Resize(64),
        T.CenterCrop(64),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
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
    "sgd": Config(lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=False),
    "adam": Config(lr=0.01, betas=(0.9, 0.999), weight_decay=0.)
}


# the learning schedular can be added here
LEARNING_POLICY = {
    "null": (
        "StepLR",
        Config(
            step_size=9999999999999,
            gamma=1
        ),
        "Null leaning policy will be applied: " \
        "keep the learning rate fixed during training."
    ),
   "cosine":(   
        "CosineAnnealingLR",   
        Config(          
            T_max=100000,
            eta_min=0.,
            last_epoch=-1,
        ),
        "cosine learning policy: T_max == epochs - 1"
    )
}








