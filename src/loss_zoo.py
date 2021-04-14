

import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_wrapper(func, reduction="mean", **kwargs):
    def wrapper(*args, **newkwargs):
        newkwargs.update(kwargs)
        loss = func(*args, **newkwargs)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            raise NotImplementedError(f"No such reduction mode: {reduction} ...")
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

class LossFunc(nn.Module):

    def __init__(self, mode: str, reduction="mean", **kwargs):
        super(LossFunc, self).__init__()
        if mode == "gen":
            self.loss_func = reduce_wrapper(
                self.criterion_gen, reduction=reduction, **kwargs
            )
        elif mode == "dis":
            self.loss_func = reduce_wrapper(
                self.criterion_dis, reduction=reduction, **kwargs
            )
        else:
            raise ValueError(f"No such mode: {mode} ...")

    def criterion_gen(self, *inputs):
        raise NotImplementedError()
    
    def criterion_dis(self, *inputs):
        raise NotImplementedError()

class BCELoss(LossFunc):

    def criterion_gen(self, outs: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(outs)

    def criterion_dis(
        self, outs_real: torch.Tensor, outs_fake: torch.Tensor
    ) -> torch.Tensor:
        assert outs_real.size(0) == outs_fake.size(0), \ 
            f"the batch size of outs_real: {outs_real.size(0)} " \ 
            f"doesnot match that of outs_fake: {outs_fake.size(0)}"
        return -F.logsigmoid(outs_real) - F.logsigmoid(1 - outs_fake)
