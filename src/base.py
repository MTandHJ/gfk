

from typing import Callable, TypeVar, Union, Optional, List, Tuple, Dict, Iterable, NoReturn
import torch
import torch.nn as nn
from collections.abc import Iterable

from .utils import AverageMeter, ProgressMeter
from .loadopts import load_dataloader
from models.base import Generator, Discriminator
from metrics.datasets import TensorDataset
from metrics.fid_score import fid_score_single
from metrics.inception_score import inception_score
from metrics.utils import load_inception


class Coach:

    def __init__(
        self, generator: Generator, 
        discriminator: Discriminator,
        inception_model: nn.Module,
        device: torch.device,
        trainloader: Iterable
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.inception_model = inception_model
        self.device = device
        self.primeloader = trainloader
        self.trainloader = iter(trainloader)
        self.loss_g = AverageMeter("Loss_G")
        self.loss_d = AverageMeter("Loss_D")
        self.progress = ProgressMeter(self.loss_g, self.loss_d)

    def save(self, path: str, postfix: str = '') -> None:
        self.generator.save(path, postfix)
        self.discriminator.save(path, postfix)

    @property
    def data(self):
        try:
            data = next(self.trainloader)
        except StopIteration:
            self.trainloader = iter(self.primeloader)
            data = next(self.trainloader)
        finally:
            data = [item.to(self.device) for item in data]
            return data

    def _unitG(
        self, 
        cur_step: int,
        batch_size: int, 
        steps_per_G: int = 1, 
        acml_per_step: int = 1
    ) -> NoReturn:

        self.generator.on()
        self.discriminator.off()
        for step in range(steps_per_G):
            self.generator.optimizer.zero_grad()

            for _ in range(acml_per_step):
                z = self.generator.sample(batch_size)
                inputs_fake = self.generator(z)
                outs_g = self.discriminator(inputs_fake)
                loss_g = self.generator.criterion(outs_g) # real...
                loss_g.backward()

                self.loss_g.update(loss_g.item(), n=batch_size, mode="mean")

            self.generator.optimizer.step()
            self.generator.ema_update(step=cur_step + step)
            self.generator.learning_policy.step()

    def _unitD(
        self, 
        cur_step: int,
        steps_per_D: int = 1, 
        acml_per_step: int = 1
    ) -> NoReturn:

        self.generator.off()
        self.discriminator.on()
        for step in range(steps_per_D):
            self.discriminator.optimizer.zero_grad()

            for _ in range(acml_per_step):
                inputs_real, _ = self.data
                batch_size = inputs_real.size(0)
                z = self.generator.sample(batch_size)
                inputs_fake = self.generator(z)
                inputs = torch.cat((inputs_real, inputs_fake), dim=0)
                outs_d = self.discriminator(inputs)
                loss_d = self.discriminator.criterion(*outs_d.chunk(2))
                loss_d.backward()

                self.loss_d.update(loss_d.item(), n=batch_size, mode="mean")

            self.discriminator.optimizer.step()
            self.discriminator.learning_policy.step()
        
    def train(
        self,
        batch_size: int,
        steps_per_G: int = 1,
        steps_per_D: int = 1,
        acml_per_step: int = 1,
        *, step: int = 8888
    ) -> Tuple[float, float]:
        """
        steps_per_G: total steps per G training procedure
        steps_per_D: total steps per D training procedure
        acml_per_step: accumulative iterations per step
        """

        # for Discriminator
        self._unitD(
            cur_step=step,
            steps_per_D=steps_per_D,
            acml_per_step=acml_per_step
        )

        # for Generator
        self._unitG(
            cur_step=step,
            batch_size=batch_size,
            steps_per_G=steps_per_G,
            acml_per_step=acml_per_step
        )

        return self.loss_g.avg, self.loss_d.avg

    def evaluate(
        self,
        dataset_type: str,
        n: int = 10000,
        batch_size: int = 16,
        n_splits: int = 1,
        need_fid: bool = True,
        need_is: bool = True
    ):

        fid_score = -1
        is_score = -1
        data = []
        for _ in range(0, n, batch_size):
            data.append(self.generator.evaluate(batch_size=batch_size).detach().cpu())
        data = torch.cat(data)
        dataset = TensorDataset(data)
        dataloader = load_dataloader(
            dataset=dataset,
            batch_size=batch_size,
        )
        
        if need_fid:
            fid_score = fid_score_single(
                dataloader=dataloader,
                dataset_type=dataset_type,
                model=self.inception_model,
                device=self.device
            )
        
        if need_is:
            is_score, is_std = inception_score(
                dataloader=dataloader,
                model=self.inception_model,
                device=self.device,
                n_splits=n_splits
            )

        return fid_score, is_score















    





