import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from datasets import mnist, mnist_m
from models.ganin import GaninModel
import utils


def train(
    model,
    epoch,
    config,
    criterion_l,
    criterion_d,
    optimizer,
    alpha,
    trainloader_m,
    trainloader_mm,
    testloader_mm,
    device='cpu',
):

    model.train()
    running_loss_total = 0

    for batch_idx, ((imgs_src, lbls_src),
                    (imgs_tgt, _)) in enumerate(zip(trainloader_m,
                                                    trainloader_mm),
                                                start=1):

        loss_total = 0

        optimizer.zero_grad()
        # source domain
        imgs_src, lbls_src = imgs_src.to(device), lbls_src.to(device)
        imgs_src = torch.cat(3 * [imgs_src], 1)

        out_l, out_d = model(imgs_src, alpha)
        loss_l_src = criterion_l(out_l, lbls_src)
        actual_d = torch.zeros(out_d.shape).to(device)
        loss_d_src = criterion_d(out_d, actual_d)

        # target domain
        imgs_tgt = imgs_tgt.to(device)

        _, out_d = model(imgs_tgt, alpha)
        actual_d = torch.ones(out_d.shape).to(device)
        loss_d_tgt = criterion_d(out_d, actual_d)

        loss_total = loss_d_src + loss_l_src + loss_d_tgt
        loss_total.backward()
        optimizer.step()

        running_loss_total += loss_total

        if batch_idx % 300 == 0:
            print(
                f"Epoch: {epoch}/{config['epochs']} Batch: {batch_idx}/{len(trainloader_m)}"
            )
            print(f"Total Loss: {running_loss_total/batch_idx}")


def test(model, testloader_mm, criterion_l, optimizer):

    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        model.eval()
        for imgs, lbls in testloader_mm:

            imgs, lbls = imgs.to(device), lbls.to(device)
            logits, _ = model(imgs, 0)  # alpha=0 for test
            # test_loss += criterion_l(logits, lbls)

            # derive which class index corresponds to max value
            preds_l = torch.max(logits, dim=1)[1]
            equals = torch.eq(preds_l,
                              lbls)  # count no. of correct class predictions
            accuracy += torch.mean(equals.float())

    print(f"Test accuracy: {accuracy / len(testloader_mm)}")
    print("\n")
