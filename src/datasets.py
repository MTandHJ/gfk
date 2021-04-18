

"""
Reference:
ajbrock
BigGAN-PyTorch
https://github.com/ajbrock/BigGAN-PyTorch
"""

from typing import NoReturn, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

import numpy as np
import os
import h5py as h5

from .config import  ROOT, HDF5, HDF5_POSFIX, NUM_WORKERS, \
                        TRANSFORMS, NORMALIZATIONS
from .utils import mkdirs



def _dataset(
    dataset_type: str, 
    mode: str = "train"
) -> torch.utils.data.Dataset:
    """
    Dataset:
    mnist: MNIST
    cifar10: CIFAR-10
    cifar100: CIFAR-100
    celeba: CelebA
    """
    try:
        transform = TRANSFORMS[dataset_type]
        transform = torchvision.transforms.Compose(transform)
    except KeyError:
        raise DatasetNotIncludeError(f"Dataset {dataset_type} or transform {transform} is not included.\n" \
                        f"Refer to the following: {_dataset.__doc__}")

    train = True if mode == "train" else False
    if dataset_type == "mnist":
        dataset = torchvision.datasets.MNIST(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    elif dataset_type == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    elif dataset_type == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    elif dataset_type == "celeba":
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(ROOT, dataset_type),
            transform=transform
        )
    return dataset


def check_hdf5(filename: str, mode: bool = 'train') -> Tuple[bool, str]:
    filename = ".".join((filename, HDF5_POSFIX))
    file_ = os.path.join(HDF5, filename)
    if os.path.exists(file_):
        with h5.File(file_, mode='a') as f:
            if f.get(mode) is None:
                print(f">>> Create group '{mode}' ...")
                f.create_group(mode)
                return False, file_
        print(f"Group '{mode}' already exists ...")
        return True, file_
    else:
        mkdirs(HDF5)
        with h5.File(file_, mode='a') as f:
            print(f">>> Create group {mode}")
            f.create_group(mode)
        return False, file_

def make_hdf5(
    dataset, file_: str, mode: str,
    batch_size: int = 256, chunks: int = 512
) -> NoReturn:
    print(f">>> Make hdf5 file for mode '{mode}' at {file_}")

    datasize = (len(dataset),)
    datashape = (dataset[0][0]).shape
    chunks = (chunks,)
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, pin_memory=False,
        num_workers=NUM_WORKERS
    )
    with h5.File(file_, "a") as f:
        fg = f[mode]
        fg.create_dataset(
            "data", 
            shape = (0,) + datashape, 
            maxshape = datasize + datashape, 
            chunks = chunks + datashape,
            dtype = "uint8"
        )
        fg.create_dataset(
            "target", 
            shape = (0,), 
            maxshape = datasize, 
            chunks = chunks,
            dtype = "int64"
        )
        for i, (data, target) in enumerate(dataloader):
            data = (255 * data).byte().numpy()
            target = target.numpy()

            csz = fg['data'].shape[0]
            bsz = data.shape[0]
            fg['data'].resize(csz + bsz, axis=0)
            fg['target'].resize(csz + bsz, axis=0)
            fg['data'][-bsz:] = data
            fg['target'][-bsz:] = target
    

class HDF5Dataset(Dataset):

    def __init__(
        self,
        file_: str,
        types: Tuple[str],
        group: str = "train",
        mv2memory: bool = False
    ):

        self.file = file_
        self.types = types
        self.group = group
        self.mv2memory = mv2memory

        with h5.File(self.file, mode='r') as f:
            fg = f[self.group]
            self.datasize = len(fg[self.types[-1]])
        
        if mv2memory:
            print(">>> Move the total data to memory ...")
            self.data = dict()
            with h5.File(self.file, mode='r') as f:
                fg = f[self.group]
                for type_ in self.types:
                    self.data[type_] = fg[type_][...]
    
    def __len__(self):
        return self.datasize
    
    def __getitem__(self, idx: int) -> List:
        if self.mv2memory:
            data = [self.data[type_][idx] for type_ in self.types]
        else:
            with h5.File(self.file, mode='r') as f:
                fg = f[self.group]
                data = [fg[type_][idx] for type_ in self.types]
        return data


class LoadHDF5Dataset(Dataset):
    TYPES = ('data', 'target')
    def __init__(
        self,
        dataset_type: str,
        mode: str = "train",
        mv2memory: bool = False
    ):
        exists, file_ = check_hdf5(dataset_type)
        if not exists:
            print("HDF5 file is not found ...")
            dataset = _dataset(
                dataset_type=dataset_type,
                mode=mode
            )
            make_hdf5(
                dataset=dataset,
                file_=file_,
                mode=mode
            )

        self.data = HDF5Dataset(
            file_, 
            LoadHDF5Dataset.TYPES,
            group=mode, mv2memory=mv2memory
        )

        self.normalizer = NORMALIZATIONS[dataset_type]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img, label = self.data[idx]

        img = self.normalizer(torch.from_numpy(img).float() / 255)
        label = int(label)
        return img, label



class LoadPrimeDataset(Dataset):

    def __init__(
        self,
        dataset_type: str,
        mode: str = 'train'
    ):

        self.data = _dataset(
            dataset_type=dataset_type,
            mode=mode
        )

        self.normalizer = NORMALIZATIONS[dataset_type]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = self.normalizer(img)
        return img, label

