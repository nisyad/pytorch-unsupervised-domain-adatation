import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from datasets import mnist, mnist_m
from models.ganin import GaninModel
from trainer import train, test
from utils import transform, helper

# Random Seed
helper.set_random_seed(seed=12345)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
config = dict(epochs=2,
              batch_size=64,
              learning_rate=2e-4,
              classes=10,
              img_size=28,
              experiment='minst-minist_m')


def main():

    model = GaninModel().to(device)

    # transforms
    transform_m = transform.get_transform(dataset="mnist")
    transform_mm = transform.get_transform(dataset="mnist_m")

    # dataloaders
    loaders_args = dict(
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    trainloader_m = mnist.fetch(data_dir="data/mnist/processed/train.pt",
                                **loaders_args)

    testloader_m = mnist.fetch(data_dir="data/mnist/processed/test.pt",
                               **loaders_args)

    trainloader_mm = mnist_m.fetch(data_dir="data/mnist_m/processed/train.pt",
                                   **loaders_args)

    testloader_mm = mnist_m.fetch(data_dir="data/mnist_m/processed/test.pt",
                                  **loaders_args)

    # criterion
    criterion_l = nn.CrossEntropyLoss()
    criterion_d = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    start_time = datetime.now()
    for epoch in range(config["epochs"]):

        alpha = helper.get_alpha(epoch, config["epochs"])
        print("alpha: ", alpha)

        train(model, epoch, config, criterion_l, criterion_d, optimizer, alpha,
              trainloader_m, trainloader_mm, testloader_mm, device)

        test(model, testloader_mm, criterion_l, optimizer, device)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Training Time for {config['epochs']} epochs: {duration}")

    return model


if __name__ == "__main__":
    model = main()
