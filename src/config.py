


# Here are some basic settings.
# It could be overwritten if you want to specify
# special configs. However, please check the correspoding
# codes in loadopts.py.



import torchvision.transforms as T
from .dict2obj import Config




ROOT = "../data"
INFO_PATH = "./infos/{method}/{dataset}-{generator}-{discriminator}/{description}"
LOG_PATH = "./logs/{method}/{dataset}-{generator}-{discriminator}/{description}"
TIMEFMT = "%m%d%H"
SAVED_FILENAME = "Generator_paras.pt" # the filename of saved model


# basic properties of inputs
MEANS = {
    "mnist": None,
    "cifar10": [0.4914, 0.4824, 0.4467],
    "cifar100": [0.5071, 0.4867, 0.4408],
    "svhn": [0.5071, 0.4867, 0.4409]
}

STDS = {
    "mnist": None,
    "cifar10": [0.2471, 0.2435, 0.2617],
    "cifar100": [0.2675, 0.2565, 0.2761],
    "svhn": [0.2675, 0.2565, 0.2761]
}

NUMCLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100
}

SHAPES = {
    "mnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32)
}

TRANSFORMS = {
    "mnist": {
        'default': T.ToTensor()
    },
    "cifar10": {
        'default': T.ToTensor()
    }
}
TRANSFORMS["cifar100"] = TRANSFORMS["cifar10"]



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
   "default": (
        "MultiStepLR",
        Config(
            milestones=[100, 105],
            gamma=0.1
        ),
        "Default leaning policy will be applied: " \
        "decay the learning rate at 100 and 105 epochs by a factor 10."
    ),
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
            T_max=200,
            eta_min=0.,
            last_epoch=-1,
        ),
        "cosine learning policy: T_max == epochs - 1"
    )
}








