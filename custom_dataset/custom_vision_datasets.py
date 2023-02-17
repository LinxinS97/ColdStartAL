import numpy as np
from sklearn.datasets import fetch_covtype
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
        kwargs = {'train': train}

        if 'svhn' in name:
            name = 'SVHN'
            kwargs = {'split': 'train' if train else 'test'}

        self.dataset = datasets.__dict__[name](root=path, download=True, transform=transform, **kwargs)
        if not train:
            self.indices = np.arange(len(self.dataset))
        else:
            self.indices = indices

    def __getitem__(self, index):
        features = torch.tensor([])
        data, target = self.dataset[int(self.indices[index])]
        if self.features is not None:
            features = self.features[index]
        return data, target, features, index

    def __len__(self):
        return len(self.indices)


class TabularDataset(Dataset):
    def __init__(self, name, indices=None):
        if name == 'covtype':
            X, y = fetch_covtype(return_X_y=True)
            self.data = X[indices]
            self.target = y[indices]
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __len__(self):
        return len(self.data)


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
    elif dataset == "cifar100":
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        inference_loader = DataLoader(
            FeatureDataset(args.data, transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize,
            ]), indices, extracted_features, name=dataset),
            batch_size=args.test_batch_size, shuffle=if_shuffle,
            num_workers=args.workers, pin_memory=False,
        )
    elif dataset in ['fashion_mnist']:
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        inference_loader = DataLoader(
            FeatureDataset(args.data, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), indices, extracted_features, name=dataset),
            batch_size=args.test_batch_size, shuffle=if_shuffle,
            num_workers=args.workers, pin_memory=False,
        )
    elif dataset == 'svhn':
        inference_loader = torch.utils.data.DataLoader(
            FeatureDataset(args.data, transforms.Compose([
                transforms.Resize(size=(32, 32)),
                # transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
                transforms.ToTensor(),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ]), indices, extracted_features, name=dataset),
            batch_size=args.test_batch_size, shuffle=if_shuffle,
            num_workers=args.workers, pin_memory=False,
        )
    elif dataset == 'forest_covertypes':
        dataset = TabularDataset('covtype', indices)
        inference_loader = DataLoader(dataset, batch_size=args.test_batch_size,
                                      shuffle=if_shuffle, num_workers=args.workers, pin_memory=False)

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
    elif dataset == "cifar100":
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
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
    elif dataset in ['fashion_mnist']:
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        train_loader = DataLoader(
            FeatureDataset(args.data, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), current_indices, extracted_features, name=dataset),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False,
        )
    elif dataset == 'svhn':
        train_loader = torch.utils.data.DataLoader(
            FeatureDataset(args.data, transforms.Compose([
                transforms.Resize(size=(32, 32)),
                transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
                transforms.ToTensor(),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ]), current_indices, extracted_features, name=dataset),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False,
        )
    elif dataset == 'forest_covertypes':
        dataset = TabularDataset('covtype', current_indices)
        train_loader = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers, pin_memory=False)
    else:
        raise NotImplementedError

    return train_loader


def get_test_loader(dataset, extracted_features, args):
    if dataset in ['cifar10']:
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
    elif dataset in ['cifar100']:
        test_loader = torch.utils.data.DataLoader(
            FeatureDataset(args.data,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                    (0.2675, 0.2565, 0.2761))
                           ]),
                           train=False, features=extracted_features, name=dataset),
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False,
        )
    elif dataset in ['fashion_mnist']:
        test_loader = torch.utils.data.DataLoader(
            FeatureDataset(args.data,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)),
                           ]),
                           train=False, features=extracted_features, name=dataset),
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False,
        )
    elif dataset == 'svhn':
        test_loader = torch.utils.data.DataLoader(
            FeatureDataset(args.data,
                           transform=transforms.Compose([
                               transforms.Resize(size=(32, 32)),
                               # transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5],
                               #                        contrast=[0.2, 1.8]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                                    (0.19803012, 0.20101562, 0.19703614))
                           ]),
                           train=False, features=extracted_features, name=dataset),
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False,
        )
    elif dataset == 'forest_covertypes':
        dataset = TabularDataset('covtype', )
        test_loader = DataLoader(dataset, batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=args.workers, pin_memory=False)

    else:
        raise NotImplementedError

    return test_loader
