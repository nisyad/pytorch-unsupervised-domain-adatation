import torch
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    def __init__(self, data, transform=None):

        self.data = data[0].unsqueeze(3)  # [N,28,28,1]
        self.data = self.data.permute(0, 3, 1, 2)
        self.labels = data[1]
        self.transform = transform

    def __len__(self):
        return len(self.data)  # 60000 for training and 10000 for test

    def __getitem__(self, idx):

        X = self.data[idx].float()
        y = self.labels[idx]

        if self.transform:
            X = self.transform(X)

        return X, y


def fetch(data_dir,
          batch_size=128,
          transform=None,
          shuffle=True,
          num_workers=1,
          pin_memory=True):

    data = torch.load(data_dir)

    dataset = MNISTDataset(data=data, transform=transform)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    return dataloader