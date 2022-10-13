import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFilter
import random
from torchvision.transforms import transforms


class FeatureDataset(Dataset):
    def __init__(self, path, transform, indices=None, features=None, train=True, name=None):
        self.features = features

        if 'cifar' in name:
            name = name.replace('cifar', 'CIFAR')
        if 'mnist' in name:
            name = 'MNIST'
        if 'fashion_mnist' in name:
            name = 'FashionMNIST'

        self.dataset = datasets.__dict__[name](root=path, train=train, download=True, transform=transform)
        if not train:
            self.indices = np.arange(len(self.dataset))
        else:
            self.indices = indices

    def __getitem__(self, index):
        features = torch.tensor([])
        data, target = self.dataset[int(self.indices[index])]
        if self.features is not None:
            features = self.features[index]
        return data, target, features, int(self.indices[index])

    def __len__(self):
        return len(self.indices)


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ImageFolderEx(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        return index, sample, target


def get_inference_loader(dataset, indices, extracted_features, args, if_shuffle=False):
    if dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])
        inference_loader = DataLoader(
            FeatureDataset(args.data, transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize,
            ]), indices, extracted_features, name=dataset),
            batch_size=args.test_batch_size, shuffle=if_shuffle,
            num_workers=args.workers, pin_memory=False,
        )
    elif dataset in ['mnist', 'fashion_mnist']:
        normalize = transforms.Normalize((0.5,), (0.5,))
        inference_loader = DataLoader(
            FeatureDataset(args.data, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), indices, extracted_features, name=dataset),
            batch_size=args.test_batch_size, shuffle=if_shuffle,
            num_workers=args.workers, pin_memory=False,
        )
    else:
        raise NotImplementedError

    return inference_loader


def get_train_loader(dataset, current_indices, extracted_features, args):

    if dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])
        train_loader = DataLoader(
            FeatureDataset(args.data, transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), current_indices, extracted_features, name=dataset),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False,
        )
    elif dataset in ['mnist', 'fashion_mnist']:
        normalize = transforms.Normalize((0.5,), (0.5,))
        train_loader = DataLoader(
            FeatureDataset(args.data, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), current_indices, extracted_features, name=dataset),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False,
        )
    else:
        raise NotImplementedError

    return train_loader


def get_test_loader(dataset, extracted_features, args):
    if dataset in ['cifar10', 'cifar100']:
        test_loader = torch.utils.data.DataLoader(
            FeatureDataset(args.data,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                    std=[0.2470, 0.2435, 0.2616])
                           ]),
                           train=False, features=extracted_features, name=dataset),
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False,
        )
    elif dataset in ['mnist', 'fashion_mnist']:
        test_loader = torch.utils.data.DataLoader(
            FeatureDataset(args.data,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]),
                           train=False, features=extracted_features, name=dataset),
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False,
        )

    else:
        raise NotImplementedError

    return test_loader
