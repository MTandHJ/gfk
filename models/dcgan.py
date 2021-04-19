


from typing import Optional, Tuple
import torch
import torch.nn as nn

from .utils import setAct, setIn, setOut



class GenBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        **cfg
   ):
        super(GenBlock, self).__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            **cfg
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = setAct(activation)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x



class DisBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        bn: bool = True,
        **cfg
    ):
        super(DisBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            **cfg
        )

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()

        self.activation = setAct(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x




class Gen(nn.Module):

    DECONV_CFG = {
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "output_padding": 0
    }

    CHANNELS = [512, 256, 128, 64]
    BLOCKS = len(CHANNELS)

    def __init__(
        self,
        dim_latent: int,
        out_shape: Tuple[int],
        activation: str = "relu"
    ):
        super(Gen, self).__init__()

        c, h, w = out_shape
        channels = Gen.CHANNELS + [c]
        in_channels = channels[:-1]
        out_channels = channels[1:]
        activations = [activation] * (Gen.BLOCKS - 1) + ['tanh']

        self.fC = in_channels[0]
        self.fH = setIn(h, blocks=Gen.BLOCKS, **Gen.DECONV_CFG)
        self.fW = setIn(w, blocks=Gen.BLOCKS, **Gen.DECONV_CFG)

        self.linear = nn.Linear(dim_latent, self.fC * self.fH * self.fW)

        blocks = []
        for i in range(Gen.BLOCKS):
            blocks.append(
                GenBlock(
                    in_channels[i],
                    out_channels[i],
                    activations[i],
                    **Gen.DECONV_CFG
                )
            )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, z):
        x = self.linear(z).view(-1, self.fC, self.fH, self.fW)
        for block in self.blocks:
            x = block(x)
        return x


class Dis(nn.Module):

    CONV_CFG = {
        "kernel_size": 4,
        "stride": 2,
        "padding": 1
    }

    CHANNELS = [64, 128, 256, 512]
    BLOCKS = len(CHANNELS)

    def __init__(self, in_shape, activation: str = "leakyrelu"):
        super(Dis, self).__init__()

        c, h, w = in_shape
        channels = [c] + Dis.CHANNELS
        in_channels = channels[:-1]
        out_channels = channels[1:]
        activations = [activation] * Dis.BLOCKS
        bns = [False] + [True] * (Dis.BLOCKS - 1)

        blocks = []
        for i in range(Dis.BLOCKS):
            blocks.append(
                DisBlock(
                    in_channels[i],
                    out_channels[i],
                    activations[i],
                    bns[i],
                    **Dis.CONV_CFG
                )
            )
        self.blocks = nn.ModuleList(blocks)

        oH = setOut(h, blocks=Dis.BLOCKS, **Dis.CONV_CFG)
        oW = setOut(w, blocks=Dis.BLOCKS, **Dis.CONV_CFG)
        self.fc = nn.Linear(
            out_channels[-1] * oH * oW,
            1
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        features = x.flatten(start_dim=1)
        logits = self.fc(features)
        return logits.squeeze()



