import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFilter, Image
import random
from typing import Callable, Optional
import os
from torchvision.transforms import transforms


class ImageNetSubset(datasets.ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            indices=None
    ):
        super(ImageNetSubset, self).__init__(root, transform=transform)
        self.indices = indices

    def __getitem__(self, index):
        path, target = self.samples[self.indices[index]]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, self.indices[index]

    def __len__(self):
        return len(self.indices)


class CIFAR100Subset(Dataset):
    def __init__(self, path, transform, indices):
        self.cifar100 = datasets.CIFAR100(root=path,
                                          download=True,
                                          train=True,
                                          transform=transform)
        self.indices = indices

    def __getitem__(self, index):
        data, target = self.cifar100[self.indices[index]]
        return data, target, self.indices[index]

    def __len__(self):
        return len(self.indices)


class CIFAR10Subset(Dataset):
    def __init__(self, path, transform, indices, features=None):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=transform)
        self.indices = indices
        self.features = features

    def __getitem__(self, index):
        features = torch.tensor([])
        data, target = self.cifar10[int(self.indices[index])]
        if self.features is not None:
            features = self.features[index]
        return data, target, features, int(self.indices[index])

    def __len__(self):
        return len(self.indices)


class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None, indices=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.indices = indices
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        if self.indices is not None:
            self.img_path = [self.img_path[i] for i in self.indices]
            self.labels = [self.labels[i] for i in self.indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index


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
    if dataset == "imagenet":
        traindir = os.path.join(args.data, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        inference_loader = DataLoader(
            ImageNetSubset(traindir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), indices),
            batch_size=args.batch_size, shuffle=if_shuffle,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "imagenet_lt":
        in_lt_train_txt = './data/ImageNet_LT_train.txt'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        inference_loader = DataLoader(
            LT_Dataset(args.data, in_lt_train_txt, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), indices),
            batch_size=args.batch_size, shuffle=if_shuffle,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                         std=[0.2673, 0.2564, 0.2762])
        inference_loader = DataLoader(
            CIFAR100Subset(args.data, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), indices),
            batch_size=args.batch_size, shuffle=if_shuffle,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])
        inference_loader = DataLoader(
            CIFAR10Subset(args.data, transforms.Compose([
                transforms.RandomResizedCrop((28, 28)),
                transforms.RandomHorizontalFlip(),
                # normalize,
                transforms.ToTensor(),
            ]), indices, extracted_features),
            batch_size=args.batch_size, shuffle=if_shuffle,
            num_workers=args.workers, pin_memory=True,
        )

    else:
        raise NotImplementedError

    return inference_loader


def get_train_loader(dataset, current_indices, extracted_features, args):
    if dataset == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = DataLoader(
            ImageNetSubset(os.path.join(args.data, 'train'),
                           transforms.Compose([
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               normalize,
                           ]), current_indices),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "imagenet_lt":
        in_lt_train_txt = './data/ImageNet_LT_train.txt'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = DataLoader(
            LT_Dataset(args.data, in_lt_train_txt,
                       transforms.Compose([
                           transforms.RandomResizedCrop(224),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           normalize,
                       ]), current_indices),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                         std=[0.2673, 0.2564, 0.2762])
        train_loader = DataLoader(
            CIFAR100Subset(args.data, transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), current_indices),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])
        train_loader = DataLoader(
            CIFAR10Subset(args.data, transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), current_indices, extracted_features),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )


    else:
        raise NotImplementedError

    return train_loader


def get_test_loader(dataset, args):
    if dataset == "imagenet" or dataset == "imagenet_lt":
        valdir = os.path.join(args.data, 'val')
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar100":
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                                       std=[0.2673, 0.2564, 0.2762])
                              ]),
                              train=False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    elif dataset == "cifar10":
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                      std=[0.2470, 0.2435, 0.2616])
                             ]),
                             train=False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    else:
        raise NotImplementedError

    return test_loader
