



import torch


class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.data = dataset

        def __len__(self):
            return len(self.data)
        def __getitem__(self, index):
            img, _ = self.data[index]
            return img

class TensorDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]