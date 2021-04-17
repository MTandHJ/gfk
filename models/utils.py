



from typing import Optional, NoReturn, Tuple, Callable
import torch.nn as nn
from functools import partial

SN_BASIC_KEYS = {
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    nn.Embedding
}

SN_EXCEPT_KEYS = set()


INIT_KEYS = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    nn.Embedding
)


def spectral_norm(
    model: nn.Module,
    name: str = 'weight',
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
    basic_keys: Optional[Tuple] = None,
    except_keys: Optional[Tuple] = None
) -> NoReturn:
    basic_keys = SN_BASIC_KEYS if basic_keys is None else basic_keys
    except_keys = SN_EXCEPT_KEYS if except_keys is None else except_keys
    keys = tuple(basic_keys - except_keys)
    print(">>> Adopting spectral norm ...")
    for name_, module in model.named_modules():
        if isinstance(module, keys):
            try:
                nn.utils.spectral_norm(
                    module=module,
                    name=name,
                    n_power_iterations=n_power_iterations,
                    eps=eps,
                    dim=dim
                )
            except AttributeError:
                print(f"Skip module {name_} as missing attrs {name} ...")


def choose_activation(activation: str) -> nn.Module:
    activations = {
        "relu": partial(nn.ReLU, inplace=True),
        "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
        "elu": partial(nn.ELU, alpha=1., inplace=True),
        "gelu": nn.GELU,
        'sigmoid': nn.Sigmoid
    }

    try:
        activation = activations[activation]
    except KeyError as e:
        print(f"No such activation {activation} ...")
        raise KeyError(e)
    return activation()



def init_weights(model, init_policy=None) -> int:
    print(">>> initialize the weights ...")
    doer: Callable
    if init_policy == "ortho":
        doer = nn.init.orthogonal_
    elif init_policy == "xavier":
        doer = nn.init.xavier_normal_
    elif init_policy == "kaming":
        doer = nn.init.kaiming_normal_
    elif init_policy == "N02":
        doer = partial(nn.init.normal_, mean=0., std=0.02)
    else:
        print(f"No such init_policy: {init_policy}, skipping it ...")
        return 0
    for module in model.modules():
        if isinstance(module, INIT_KEYS):
            try:
                doer(module.weight)
                module.bias.data.fill_(0.)
            except AttributeError:
                continue
    return 1


def down(
    in_size, blocks, 
    kernel_size, stride, padding, 
    dilation=1
):
    diff = in_size + 2 * padding - dilation * (kernel_size - 1) - 1 + stride
    out_size = diff // stride
    if blocks <= 1:
        return out_size
    else:
        return down(
            out_size, kernel_size, stride, padding,
            blocks-1, dilation
        )

def up(
    in_size, blocks, 
    kernel_size, stride, padding,
    dilation=1, output_padding=0
):
    assert isinstance(blocks, int), "blocks should be integer"
    out_size = (in_size - 1) * stride - 2 * padding + dilation * (kernel_size -1) + \
            output_padding + 1
    if blocks <= 1:
        return out_size
    else:
        return up(
            out_size, kernel_size, stride, padding,
            blocks-1, dilation, output_padding
        )

def setIn(
    out_size, blocks, 
    kernel_size, stride, padding,
    dilation=1, output_padding=0
):
    left = 1
    right = out_size
    while left < right:
        in_ = (left + right) // 2
        out_ = up(in_, kernel_size, stride, padding, blocks, dilation, output_padding)
        if out_ < out_size:
            left = in_ + 1
        else:
            right = in_
    out_ = up(left, kernel_size, stride, padding, blocks, dilation, output_padding)
    if out_ != out_size:
        raise ValueError("No suitable in_size exists ...")
    return left

setOut = down

