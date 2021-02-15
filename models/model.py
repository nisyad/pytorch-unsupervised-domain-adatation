import torch.nn as nn
from torch.autograd import Function


class UDAModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(7 * 7 * 32, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 10),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(7 * 7 * 32, 100),
            nn.Linear(100, 1),
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.shape[0], 3, 28, 28)
        features = self.feature_extractor(input_data)
        features = features.view(-1, 7 * 7 * 32)
        label = self.label_classifier(features)
        reverse_features = ReverseGradientLayer.apply(features, alpha)
        domain = self.domain_classifier(reverse_features)

        return label, domain


class ReverseGradientLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grad_output):

        return -ctx.alpha * grad_output, None
