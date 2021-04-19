

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunc(nn.Module):

    def __init__(self, mode: str, reduction: str = "mean"):
        super(LossFunc, self).__init__()
        self.reduction = reduction
        if mode == "gen":
            self.loss_func = self.criterion_gen
        elif mode == "dis":
            self.loss_func = self.criterion_dis
        else:
            raise ValueError(f"No such mode: '{mode}' ...")
    
    def _reduce_proxy(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"No such mode: '{self.reduction}' ...")

    def criterion_gen(self, *inputs):
        raise NotImplementedError()
    
    def criterion_dis(self, *inputs):
        raise NotImplementedError()

    def forward(self, *inputs, **kwargs):
        return self.loss_func(*inputs, **kwargs)


class BCELoss(LossFunc):
    """
    generator:
        $\min -\log \sigma(D(G(z)))$
    discriminator:
        $\min -\log \sigma(D(x)) - \log(1 - \sigma(D(G(z))))$
    sigma is sigmoid here.
    """
    def __init__(self, mode: str, reduction="mean"):
        super(BCELoss, self).__init__(mode=mode, reduction=reduction)

        self._loss = nn.BCELoss(reduction=reduction)
        self._preprocessing = nn.Sigmoid()

    def criterion_gen(self, outs: torch.Tensor) -> torch.Tensor:
        labels = torch.ones_like(outs).to(outs.device)
        outs = self._preprocessing(outs)
        return self._loss(outs, labels)

    def criterion_dis(
        self, outs_real: torch.Tensor, outs_fake: torch.Tensor
    ) -> torch.Tensor:
        labels_real = torch.ones_like(outs_real).to(outs_real.device)
        labels_fake = torch.zeros_like(outs_fake).to(outs_fake.device)
        outs_real = self._preprocessing(outs_real)
        outs_fake = self._preprocessing(outs_fake)
        loss1 = self._loss(outs_real, labels_real)
        loss2 = self._loss(outs_fake, labels_fake)
        return loss1 + loss2


class HingeLoss(LossFunc):
    """
    generator:
        $\min -D(G(z))$
    discriminator:
        $\min \max(0, 1 - D(x)) + \max(0, 1 + D(G(z)))$
    """
    def criterion_gen(self, outs: torch.Tensor) -> torch.Tensor:
        return self._reduce_proxy(-outs)

    def criterion_dis(
        self, outs_real: torch.Tensor, outs_fake: torch.Tensor
    ) -> torch.Tensor:
        loss_real = self._reduce_proxy(F.relu(1 - outs_real))
        loss_fake = self._reduce_proxy(F.relu(1 + outs_fake))
        return loss_real + loss_fake


class WLoss(LossFunc):
    """
    WGAN Loss:
    generator:
        $\min -D(G(z))$
    discriminator:
        $\min D(x) - D(G(z))$
    """
    def criterion_gen(self, outs: torch.Tensor) -> torch.Tensor:
        return self._reduce_proxy(-outs)

    def criterion_dis(
        self, outs_real: torch.Tensor, outs_fake: torch.Tensor
    ) -> torch.Tensor:
        return self._reduce_proxy(outs_fake) - self._reduce_proxy(outs_real)


class LeastSquaresLoss(LossFunc):
    """
    generator:
        $\min (D(G(z)) - 1)^2 / 2$
    discriminator:
        $\min [(D(x) - 1)^2 + (D(G(z)))^2] / 2
    """
    def criterion_gen(self, outs: torch.Tensor) -> torch.Tensor:
        return self._reduce_proxy((outs - 1).pow(2)) / 2

    def criterion_dis(
        self, outs_real: torch.Tensor, outs_fake: torch.Tensor
    ) -> torch.Tensor:
        assert outs_real.size(0) == outs_fake.size(0), \
            f"the batch size of outs_real: {outs_real.size(0)} " \
            f"doesnot match that of outs_fake: {outs_fake.size(0)}"
        loss_real = self._reduce_proxy((outs_real - 1).pow(2))
        loss_fake = self._reduce_proxy(outs_fake.pow(2))
        return (loss_real + loss_fake) / 2

