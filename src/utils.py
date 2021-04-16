


from typing import Optional, Any, Union, Dict, NoReturn
import torch
import torch.nn as nn
import numpy as np
import random
import os, sys, pickle
from .config import SAVED_FILENAME
from freeplot.base import FreePlot



class AverageMeter:

    def __init__(self, name: str, fmt: str = ".5f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1, mode: str = "mean") -> None:
        self.val = val
        self.count += n
        if mode == "mean":
            self.sum += val * n
        elif mode == "sum":
            self.sum += val
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} Avg:{avg:{fmt}}"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, *meters: AverageMeter, prefix: str = ""):
        self.meters = list(meters)
        self.prefix = prefix

    def display(self, *, step: int = 8888) -> None:
        entries = [self.prefix + f"[Step: {step:<4d}]"]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def add(self, *meters: AverageMeter) -> None:
        self.meters += list(meters)

    def step(self) -> None:
        for meter in self.meters:
            meter.reset()


def imagemeter(*imgs):
    rows = len(imgs)
    imgs = [
        img.clone().detach().cpu().numpy().transpose((0, 2, 3, 1))
        for img in imgs
    ]
    cols = imgs[0].shape[0]
    fp = FreePlot((rows, cols), (cols, rows), dpi=100)
    for row in range(rows):
        for col in range(cols):
            fp.imageplot(imgs[row][col], index=row * cols + col)
    return fp.fig


def gpu(*models: nn.Module) -> torch.device:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model in models:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        else:
            model.to(device)
    return device

def mkdirs(*paths: str) -> None:
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

def readme(path: str, opts: "parser", mode: str = "w") -> None:
    """
    opts: the argparse
    """
    import time
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S")
    filename = path + "/README.md"
    s = "- {0[0]}:  {0[1]}\n"
    info = "\n## {0}".format(time_)
    for item in opts._get_kwargs():
        info += s.format(item)
    with open(filename, mode, encoding="utf8") as fh:
        fh.write(info)

# load model's parameters
def load(
    model: nn.Module, 
    path: str, 
    device: torch.device,
    filename: str = SAVED_FILENAME,
    strict: bool = True, 
    except_key: Optional[str] = None
) -> None:

    filename = os.path.join(path, filename)
    if str(device) =="cpu":
        state_dict = torch.load(filename, map_location="cpu")
        
    else:
        state_dict = torch.load(filename)
    if except_key is not None:
        except_keys = list(filter(lambda key: except_key in key, state_dict.keys()))
        for key in except_keys:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

# save the checkpoint
def save_checkpoint(path, state_dict: Dict) -> None:
    path = path + "/model-optim-lr_sch-step.tar"
    torch.save(
        state_dict,
        path
    )

# load the checkpoint
def load_checkpoint(path, models: Dict) -> int:
    path = path + "/model-optim-lr_sch-step.tar"
    checkpoints = torch.load(path)
    for key, model in models.items():
        checkpoint = checkpoints[key]
        model.load_state_dict(checkpoint)
    step = checkpoints['step']
    return step

def set_seed(seed: int) -> None:
    from torch.backends import cudnn
    if seed == -1:
        seed = random.randint(0, 1024)
        print(f"Choose seed randomly: {seed}")
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        cudnn.benchmark, cudnn.deterministic = False, True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def export_pickle(data: Dict, filename: str) -> NoReturn:
    print(">>> Export File ...")
    fh = None
    try:
        fh = open(filename, "wb")
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)
    except (EnvironmentError, pickle.PicklingError) as err:
        raise ExportError(f"Export Error: {err}")
    finally:
        if fh is not None:
            fh.close()

def import_pickle(filename: str) -> NoReturn:
    print(">>> Import File ...")
    fh = None
    try:
        fh = open(filename, "rb")
        return pickle.load(fh)
    except (EnvironmentError, pickle.UnpicklingError) as err:
        raise ImportError(f"Import Error: {err}")
    finally:
        if fh is not None:
            fh.close()

# caculate the lp distance along the dim you need,
# dim could be tuple or list containing multi dims.
def distance_lp(
    x: torch.Tensor, 
    y: torch.Tensor, 
    p: Union[int, float, str], 
    dim: Optional[int] = None
) -> torch.Tensor:
    return torch.norm(x-y, p, dim=dim)
