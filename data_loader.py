import torch
from torch.utils.data import Dataset, DataLoader

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])


class MNISTMDataset(Dataset):
    def __init__(self, data, transform=None):

        self.data = data[0].permute(0, 3, 1, 2)
        self.labels = data[1].float()
        self.transform = transform

    def __len__(self):
        return len(self.data)  # 60000 for training and 10000 for test

    def __getitem__(self, idx):

        X = self.data[idx].float()
        y = self.labels[idx]

        if self.transform:
            X = self.transform(X)

        return X, y


def fetch(data_dir, batch_size=128, shuffle=True, transform=None):

    data = torch.load(data_dir)

    dataset = MNISTMDataset(data=data, transform=transform)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0)

    return dataloader
