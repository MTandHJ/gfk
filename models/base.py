

from typing import Callable, TypeVar, Any, Union, Optional, List, Tuple, Dict, Iterable, cast
import torch
import torch.nn as nn
import os
import copy


T = TypeVar('T')

class GDtor(nn.Module):

    def __init__(
        self, arch: nn.Module, 
        device: torch.device,
        criterion: Callable,
        optimizer: torch.optim.Optimizer, 
        learning_policy: "learning rate policy"
    ):
        super(GDtor, self).__init__()

        self.arch = arch
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_policy = learning_policy

    def on(self) -> None:
        self.arch.train()
        self.arch.requires_grad_(True)

    def off(self) -> None:
        self.arch.eval()
        self.arch.requires_grad_(False)

    def save(self, path: str, postfix: str = "") -> None:
        torch.save(self.arch.state_dict(), 
            os.path.join(path, f"{self.__class__.__name__}{postfix}_paras.pt"))

    def state_dict(self, destination=None, prefix='', keep_vars=False) -> Dict:
        destination = super(GDtor, self).state_dict(
            destination, prefix, keep_vars
        )
        destination['optimizer'] = self.optimizer.state_dict()
        destination['learning_policy'] = self.learning_policy.state_dict()

        return destination

    def load_state_dict(self, state_dict: Dict, strict: bool = True):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.learning_policy.load_state_dict(state_dict['learning_policy'])
        del state_dict['optimizer'], state_dict['learning_policy']
        return super(GDtor, self).load_state_dict(state_dict, strict)

    def forward(self, *inputs):
        return self.arch(*inputs)



class Generator(GDtor):

    def __init__(
        self, arch: nn.Module,
        device: torch.device,
        sampler: Callable,
        dim_latent: int, 
        criterion: Callable, 
        optimizer: torch.optim.Optimizer, 
        learning_policy: "learning rate policy",
        ema: bool = True,
        mom: float = 0.9999,
        warmup_steps: int = 0,
    ):
        super(Generator, self).__init__(
            arch=arch, device=device,
            criterion=criterion,
            optimizer=optimizer,
            learning_policy=learning_policy
        )

        if isinstance(dim_latent, Iterable):
            self.dim_latent = list(dim_latent)
        else:
            self.dim_latent = [dim_latent]
        
        self.sampler = sampler
        
        self.ema = ema
        self.mom = mom
        self.warmup_steps = warmup_steps
        if ema:
            # TODO: I'm not sure the following works all the time.
            print(">>> Adopting exponential moving average (EMA) ...")
            self.shadow = copy.deepcopy(arch)
            self.shadow.requires_grad_(False)
            self.shadow.eval()
        else:
            self.shadow = arch
    
    def save(self, path: str, postfix: str = "") -> None:
        torch.save(self.shadow.state_dict(), 
            os.path.join(path, f"{self.__class__.__name__}{postfix}_paras.pt"))
        if self.ema:
            postfix = "_prime" + postfix
            torch.save(self.shadow.state_dict(), 
                os.path.join(path, f"{self.__class__.__name__}{postfix}_paras.pt"))

    @torch.no_grad()
    def ema_update(self, step: int):
        if not self.ema:
            return 0
        mom = 0. if step < self.warmup_steps else self.mom
        for key, source in self.arch.state_dict().items():
            target = self.shadow.state_dict()[key]
            data = target.data * mom + source.data * (1 - mom)
            target.data.copy_(data)
    
    def sample(self, batch_size: int) -> None:
        size = [batch_size] + self.dim_latent
        return self.sampler(size).to(self.device)
    
    @torch.no_grad()
    def evaluate(
        self, z: Optional[torch.Tensor] = None, batch_size: int = 10
    ) -> torch.Tensor:
        if z is None:
            z = self.sample(batch_size)
        else:
            z = z.to(self.device)
        self.shadow.eval()
        outs = self.shadow(z)
        return outs


class Discriminator(GDtor):

    def __init__(
        self, arch: nn.Module, 
        device: torch.device,
        criterion: Callable,
        optimizer: torch.optim.Optimizer, 
        normalizer: Callable[[T], T],
        augmentor: Callable[[T], T],
        learning_policy: "learning rate policy"
    ):
        super(Discriminator, self).__init__(
            arch=arch, device=device, 
            criterion=criterion, 
            optimizer=optimizer,
            learning_policy=learning_policy
        )

        self.normalizer = normalizer
        self.augmentor = augmentor

    def forward(self, imgs: torch.Tensor, *others):
        imgs = self.augmentor(imgs)
        imgs = self.normalizer(imgs)
        return self.arch(imgs, *others)

 




