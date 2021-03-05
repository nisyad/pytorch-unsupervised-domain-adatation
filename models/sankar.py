import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, num_filters_disc=64):
        super().__init__()

        self.num_filters_disc = num_filters_disc

        self.feature = nn.Sequential(
            nn.Conv2d(3, self.num_filters_disc, 5, stride=1, padding=0),
            nn.ReLU(), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.num_filters_disc,
                      self.num_filters_disc,
                      5,
                      stride=1,
                      padding=0), nn.ReLU(), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.num_filters_disc,
                      2 * self.num_filters_disc,
                      5,
                      stride=1,
                      padding=0), nn.ReLU())

    def forward(self, input):
        output = self.feature(input)

        return output.view(-1, 2 * self.num_filters_disc)


class ClassifierNet(nn.Module):
    def __init__(self, num_filters_disc=64, n_classes=10):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(2 * num_filters_disc, 2 * num_filters_disc),
            nn.ReLU(),
            nn.Linear(2 * num_filters_disc, n_classes),
        )

    def forward(self, x):
        output = self.classifier(x)
        return output


class Generator(nn.Module):
    def __init__(self, n_dim, z_dim, n_classes, num_filters_gen=64):
        super().__init__()

        self.n_dim = n_dim
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.num_filters_gen = num_filters_gen

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(
                self.z_dim + self.n_dim + n_classes + 1,
                self.num_filters_gen * 8,
                2,
                1,
                0,
                bias=False,
            ),
            nn.BatchNorm2d(self.num_filters_gen * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_filters_gen * 8,
                               self.num_filters_gen * 4,
                               4,
                               2,
                               1,
                               bias=False),
            nn.BatchNorm2d(self.num_filters_gen * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_filters_gen * 4,
                               self.num_filters_gen * 2,
                               4,
                               2,
                               1,
                               bias=False),
            nn.BatchNorm2d(self.num_filters_gen * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_filters_gen * 2,
                               self.num_filters_gen,
                               4,
                               2,
                               1,
                               bias=False),
            nn.BatchNorm2d(self.num_filters_gen),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_filters_gen, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        # batch_size = input.shape[0]
        input = input.view(-1, self.z_dim + self.n_dim + self.n_classes + 1, 1,
                           1)
        # noise = torch.randn(batch_size, self.z_dim, 1, 1)
        # output = self.gen(torch.cat((input, noise), 1))
        return self.gen(input)


class Discriminator(nn.Module):
    def __init__(self, n_classes, num_filters_disc=64):
        super().__init__()

        self.n_classes = n_classes
        self.num_filters_disc = num_filters_disc

        self.disc = nn.Sequential(
            nn.Conv2d(3, self.num_filters_disc, 3, 1, 1),
            nn.BatchNorm2d(self.num_filters_disc),
            nn.LeakyReLU(0.2, ),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.num_filters_disc, self.num_filters_disc * 2, 3, 1,
                      1),
            nn.BatchNorm2d(self.num_filters_disc * 2),
            nn.LeakyReLU(0.2, ),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.num_filters_disc * 2, self.num_filters_disc * 4, 3,
                      1, 1),
            nn.BatchNorm2d(self.num_filters_disc * 4),
            nn.LeakyReLU(0.2, ),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.num_filters_disc * 4, self.num_filters_disc * 2, 3,
                      1, 1),
            nn.BatchNorm2d(self.num_filters_disc * 2),
            nn.LeakyReLU(0.2, ),
            nn.MaxPool2d(4, 4),
        )

        # Domain classifier - binary
        self.classifier_d = nn.Sequential(
            nn.Linear(self.num_filters_disc * 2, 1),
            nn.Sigmoid(),
        )

        self.classifier_c = nn.Sequential(
            nn.Linear(self.num_filters_disc * 2, self.n_classes))

    def forward(self, input):
        output = self.disc(input)
        output = output.view(-1, self.num_filters_disc * 2)
        output_d = self.classifier_d(output)
        output_c = self.classifier_c(output)

        return output_d.view(-1), output_c
