

from typing import Callable, TypeVar, Union, Optional, List, Tuple, Dict, Iterable
import torch
import torch.nn as nn
from collections.abc import Iterable

from .utils import AverageMeter, ProgressMeter
from models.base import Generator, Discriminator


class Coach:

    def __init__(
        self, generator: Generator, 
        discriminator: Discriminator,
        device: torch.device, normalizer: Callable, 
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.normalizer = normalizer
        self.loss_g = AverageMeter("Loss_G")
        self.loss_d = AverageMeter("Loss_D")
        self.validity = AverageMeter("Validity")
        self.progress = ProgressMeter(self.loss_g, self.loss_d, self.validity)

    def save(self, path: str, postfix: str = '') -> None:
        self.generator.save(path, postfix)
        self.discriminator.save(path, postfix)

    def train(self, trainloader: Iterable, *, epoch: int = 8888) -> Tuple[float, float, float]:
        self.progress.step() # reset the meter
        for inputs_real, _ in trainloader:
            batch_size = inputs_real.size(0)
            labels_real = torch.ones(batch_size).to(self.device)
            labels_fake = torch.zeros(batch_size).to(self.device)
            inputs_real = inputs_real.to(self.device)
            
            # generator part
            self.generator.train()
            self.discriminator.eval()
            z = self.generator.sample(batch_size)
            inputs_fake = self.generator(z)
            outs_g = self.discriminator(self.normalizer(inputs_fake))
            loss_g = self.generator.criterion(outs_g, labels_real) # real...

            # update the generator
            self.generator.optimizer.zero_grad()
            loss_g.backward()
            self.generator.optimizer.step()

            # discriminator part
            self.generator.eval()
            self.discriminator.train()
            inputs = torch.cat((inputs_real, inputs_fake.detach()), dim=0)
            labels = torch.cat((labels_real, labels_fake), dim=0)
            outs_d = self.discriminator(self.normalizer(inputs))
            loss_d = self.discriminator.criterion(outs_d, labels)

            # update the discriminator
            self.discriminator.optimizer.zero_grad()
            loss_d.backward()
            self.discriminator.optimizer.step()

            # log
            validity = (outs_d.round() == labels).sum().item()
            self.loss_g.update(loss_g.item(), n=batch_size, mode="mean")
            self.loss_d.update(loss_d.item(), n=batch_size, mode="mean")
            self.validity.update(validity, n=batch_size * 2, mode="sum")
        
        self.progress.display(epoch=epoch)
        self.generator.learning_policy.step()
        self.discriminator.learning_policy.step()

        return self.loss_g.avg, self.loss_d.avg, self.validity.avg













    





