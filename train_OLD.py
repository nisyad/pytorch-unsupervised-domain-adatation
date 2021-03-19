import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from datasets import mnist, mnist_m
from models.ganin import GaninModel

# Set Random Seeds
torch.backends.cudnn.deterministic = True
np.random.seed(hash("setting random seeds improves") % 2**32 - 1)
torch.manual_seed(hash("reproducibility by removing randomness") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Pytorch performance tuninng guide - NVIDIA
torch.backends.cudnn.benchmark = True  # speeds up convolution operations

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# HYPERPARAMETERS
IMG_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 2
LR = 2e-4

# # MNIST
# transform_m = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, ), (0.5, ))])

# trainset_m = datasets.MNIST("data/mnist",
#                             train=True,
#                             download=True,
#                             transform=transform_m)
# trainloader_m = torch.utils.data.DataLoader(trainset_m,
#                                             batch_size=BATCH_SIZE,
#                                             shuffle=True)

# testset_m = datasets.MNIST("data/mnist",
#                            train=False,
#                            download=True,
#                            transform=transform_m)

# testloader_m = torch.utils.data.DataLoader(testset_m,
#                                            batch_size=BATCH_SIZE,
#                                            shuffle=True)

# # MNIST-M
# transform_mm = transforms.Compose(
#     [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# DATA_DIR = "data/mnist_m/processed/"

# trainloader_mm = data_loader.fetch(
#     data_dir=os.path.join(DATA_DIR, 'mnist_m_train.pt'),
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     transform=transform_mm,
# )

# testloader_mm = data_loader.fetch(
#     data_dir=os.path.join(DATA_DIR, 'mnist_m_test.pt'),
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     transform=transform_mm,
# )

# MNIST -- [N, 28, 28]
transform_m = transforms.Compose([transforms.Normalize((0.5, ), (0.5, ))])

DATA_DIR = "data/mnist/processed/"
trainloader_m = mnist.fetch(
    data_dir=os.path.join(DATA_DIR, 'train.pt'),
    batch_size=BATCH_SIZE,
    transform=transform_m,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

testloader_m = mnist.fetch(
    data_dir=os.path.join(DATA_DIR, 'test.pt'),
    batch_size=BATCH_SIZE,
    transform=transform_m,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

# MNIST-M -- [N, 28, 28, 3]
transform_mm = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

DATA_DIR = "data/mnist_m/processed/"
trainloader_mm = mnist_m.fetch(
    data_dir=os.path.join(DATA_DIR, 'train.pt'),
    batch_size=BATCH_SIZE,
    transform=transform_mm,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

testloader_mm = mnist_m.fetch(
    data_dir=os.path.join(DATA_DIR, 'test.pt'),
    batch_size=BATCH_SIZE,
    transform=transform_mm,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

print(next(iter(trainloader_m))[0].shape)

# INILIALIZE NET
# net = GaninModel().to(device)

# # SET CRITERION and OPTIMIZER
# criterion_l = nn.CrossEntropyLoss()
# criterion_d = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(net.parameters(), lr=LR)

# num_batches = min(len(trainloader_m), len(trainloader_mm))  # ~60000/batch_size
# print("ggggg...No. of Batches: ", num_batches)

# # DEVICE DETAILS
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Allocated: (GB)')
#     print('Allocated: ', round(torch.cuda.memory_allocated(0) / 1024**3, 1))
#     print('Cached: ', round(torch.cuda.memory_reserved(0) / 1024**3, 1))

# # TRAINING
# test_accuracy = []
# start_time = datetime.now()

# for epoch in range(EPOCHS):

#     running_loss_total = 0
#     running_loss_l = 0
#     running_loss_d = 0

#     dataiter_mm = iter(trainloader_mm)
#     dataiter_m = iter(trainloader_m)
#     alpha = (2 / (1 + np.exp(-10 * ((epoch + 0.0) / EPOCHS)))) - 1
#     print(f"alpha: {alpha}")

#     net.train()
#     for batch in range(1, num_batches + 1):
#         loss_total = 0
#         loss_d = 0
#         loss_l = 0

#         optimizer.zero_grad()
#         #for source domain
#         imgs, lbls = dataiter_m.next()
#         imgs, lbls = imgs.to(device), lbls.to(device)
#         imgs = torch.cat((imgs, imgs, imgs), 1)

#         # with torch.cuda.amp.autocast():
#         out_l, out_d = net(imgs, alpha)
#         loss_l_src = criterion_l(out_l, lbls)
#         actual_d = torch.zeros(out_d.shape).to(device)
#         loss_d_src = criterion_d(out_d, actual_d)

#         #for target domain
#         imgs, lbls = dataiter_mm.next()
#         imgs = imgs.to(device)

#         # with torch.cuda.amp.autocast():
#         _, out_d = net(imgs, alpha)
#         actual_d = torch.ones(out_d.shape).to(device)
#         loss_d_tgt = criterion_d(out_d, actual_d)

#         loss_total = loss_d_src + loss_l_src + loss_d_tgt
#         loss_total.backward()
#         optimizer.step()

#         running_loss_total += loss_total
#         running_loss_d += loss_d_src + loss_d_tgt
#         running_loss_l += loss_l_src

#         if batch % 300 == 0:
#             print(f"Epoch: {epoch}/{EPOCHS} Batch: {batch}/{num_batches}")
#             print(f"Total Loss: {running_loss_total/batch}")
#         #   print(f"Label Loss: {running_loss_l/batch}")
#         #   print(f"Domain Loss: {running_loss_d/batch}")

#     net.eval()
#     test_loss = 0
#     accuracy = 0

#     with torch.no_grad():
#         net.eval()
#         for imgs, lbls in testloader_mm:
#             imgs, lbls = imgs.to(device), lbls.to(device)
#             # print(imgs.shape)
#             # print(lbls.shape)

#             logits, _ = net(imgs, alpha=0)
#             # print(logits.shape)
#             test_loss += criterion_l(logits, lbls)

#             # derive which class index corresponds to max value
#             preds_l = torch.max(
#                 logits,
#                 dim=1)[1]  # [1]: indices(class) corresponding to max values
#             equals = torch.eq(preds_l,
#                               lbls)  # count no. of correct class predictions
#             accuracy += torch.mean(equals.float())

#     test_accuracy.append(accuracy / len(testloader_mm))
#     print(f"Test accuracy: {accuracy / len(testloader_mm)}")
#     print("\n")

# end_time = datetime.now()
# duration = end_time - start_time
# print(f"Training Time for {EPOCHS} epochs: {duration}")
