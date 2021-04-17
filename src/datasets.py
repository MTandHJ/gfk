

from typing import NoReturn, Tuple
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


def check_hdf5(filename: str):
    filename = ".".join((filename, HDF5_POSFIX))
    file_ = os.path.join(HDF5, filename)
    mkdirs(HDF5)
    if os.path.exists(file_):
        return True, file_
    else:
        return False, file_

def make_hdf5(
    dataset, filename: str, 
    batch_size: int = 256, chunks: int = 256
) -> NoReturn:
    exists, file_ = check_hdf5(filename)
    if exists:
        OverwriteError_ = type("OverwriteError", (Exception,), dict())
        raise OverwriteError_(f"file_ already exists ...")
    
    print(f">>> Make hdf5 file for {filename} at {file_}")

    datasize = (len(dataset),)
    datashape = (dataset[0][0]).shape
    chunks = (chunks,)
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, pin_memory=False,
        num_workers=NUM_WORKERS
    )
    with h5.File(file_, "a") as f:
        f.create_dataset(
            "data", 
            shape = (0,) + datashape, 
            maxshape = datasize + datashape, 
            chunks = chunks + datashape,
            dtype = "uint8"
        )
        f.create_dataset(
            "target", 
            shape = (0,), 
            maxshape = datasize, 
            chunks = chunks,
            dtype = "int64"
        )
        for i, (data, target) in enumerate(dataloader):
            data = (255 * data).byte().numpy()
            target = target.numpy()

            csz = f['data'].shape[0]
            bsz = data.shape[0]
            f['data'].resize(csz + bsz, axis=0)
            f['target'].resize(csz + bsz, axis=0)
            f['data'][-bsz:] = data
            f['target'][-bsz:] = target
    

def load_hdf5(filename: str) -> Tuple[np.ndarray]:
    exists, file_ = check_hdf5(filename)
    if not exists:
        raise FileNotFoundError(f"No such file exists {filename} ...")
    with h5.File(file_, "r") as f:
        data, target = f['data'][...], f['target'][...]
    data = data.astype(np.float32) / 255.
    return data, target


class LoadDataset(Dataset):

    def __init__(
        self,
        dataset_type: str,
        mode: str = "train"
    ):
        filename = "_".join((dataset_type, mode))
        try:
            self.data, self.target = load_hdf5(filename)
        except FileNotFoundError:
            print("HDF5 file is not found ...")
            dataset = _dataset(
                dataset_type=dataset_type,
                mode=mode
            )
            make_hdf5(
                dataset=dataset,
                filename=filename
            )
            self.data, self.target = load_hdf5(filename)

        self.normalizer = NORMALIZATIONS[dataset_type]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[index]
        img = self.normalizer(torch.from_numpy(img))

        label = self.target[index]
        label = int(label)
        return img, label

