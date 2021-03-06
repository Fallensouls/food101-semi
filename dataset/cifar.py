import logging

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms.transforms import Resize
from .randaugment import RandAugmentMC
from .food101 import get_food101_data
# from .n_food101 import get_food101_n_data
from .uecfood import get_uecfood_data

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
food101_mean = (0.54614421,0.44465557,0.34460956)
food101_std =  (0.26745502,0.27089352,0.27548077)
uecfood100_mean = (0.59842609, 0.49229521, 0.36115813)
uecfood100_std = (0.23920553, 0.24704192, 0.26638424)
uecfood256_mean = (0.60702168, 0.49392965, 0.35589468)
uecfood256_std = (0.23429272, 0.24887561, 0.26953048)

normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(root, num_labeled):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        base_dataset.targets, num_labeled, num_classes=10)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFix(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)
    logger.info("Dataset: CIFAR10")
    logger.info(f"Labeled examples: {len(train_labeled_idxs)}"
                f" Unlabeled examples: {len(train_unlabeled_idxs)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(root, num_labeled):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        base_dataset.targets, num_labeled, num_classes=100)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFix(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    logger.info("Dataset: CIFAR100")
    logger.info(f"Labeled examples: {len(train_labeled_idxs)}"
                f" Unlabeled examples: {len(train_unlabeled_idxs)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_food101(root, labeledPercentage):
    transform_labeled = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=64,
                              padding=int(64*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=food101_mean, std=food101_std)])
    LabeledSet,UnlabeledSet,TestSet = get_food101_data(root,labeledPercentage,transform_labeled,TransformFix(mean=food101_mean, std=food101_std))
    logger.info("Dataset: food101")
    return LabeledSet, UnlabeledSet, TestSet

def get_uecfood100(root, labeledPercentage):
    transform_labeled = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=64,
                              padding=int(64*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=uecfood100_mean, std=uecfood100_std)])
    LabeledSet,UnlabeledSet,TestSet = get_uecfood_data(root,labeledPercentage,transform_labeled,TransformFix(mean=uecfood100_mean, std=uecfood100_std))
    logger.info("Dataset: UECFOOD100")
    return LabeledSet, UnlabeledSet, TestSet

def get_uecfood256(root, labeledPercentage):
    transform_labeled = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=64,
                              padding=int(64*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=uecfood256_mean, std=uecfood256_std)])
    LabeledSet,UnlabeledSet,TestSet = get_uecfood_data(root,labeledPercentage,transform_labeled,TransformFix(mean=uecfood256_mean, std=uecfood256_std))
    logger.info("Dataset: UECFOOD256")
    return LabeledSet, UnlabeledSet, TestSet

def x_u_split(labels,
              num_labeled,
              num_classes):
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    np.random.shuffle(labeled_idx)
    return np.array(labeled_idx), np.array(unlabeled_idx)


class TransformFix(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        # self.byol = get_simclr_data_transforms([64,64], mean, std)

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'food101': get_food101,
                   'uecfood100': get_uecfood100,
                   'uecfood256': get_uecfood256}
