from torchvision import transforms


def get_transform(dataset):
    if dataset == "mnist":
        return transforms.Compose([transforms.Normalize((0.5, ), (0.5, ))])

    elif dataset == "minist_m":
        return transforms.Compose(
            [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
