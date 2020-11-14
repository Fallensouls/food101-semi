from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import random
from PIL import Image
import time
import cv2
import numpy as np
import re
import os

class TrainDataset(Dataset):
    def __init__(self,path,list,transform,label):
        self.list = list
        self.len = len(list)
        self.path = path
        self.transform = transform
        self.label = label

    def __getitem__(self,index):
        pic_path = os.path.join(self.path,'images',self.list[index].replace("\n","")+'.jpg')
        img = Image.open(pic_path).convert('RGB')
        img = self.transform(img)
        return np.array(img),self.label[index]

    def __len__(self):
        return self.len

class TestDataset(Dataset):
    def __init__(self,path,list,transform,label):
        self.list = list
        self.len = len(list)
        self.path = path
        self.resize = transforms.Resize((32,32))
        self.transform = transform
        self.label = label

    def __getitem__(self,index):
        pic_path = os.path.join(self.path,'images',self.list[index].replace("\n","")+'.jpg')
        img = Image.open(pic_path).convert('RGB')
        img = self.transform(img)
        return np.array(img),self.label[index]

    def __len__(self):
        return self.len


def get_food101_n_data(path, transform, n_task):
    i = 0
    class_label= {}
    with open(os.path.join(path,'classes.txt'), 'r') as f:
        line = f.readline()
        while(line):
            class_label[line.replace("\n","")] = i
            i+=1
            line = f.readline()
    with open(os.path.join(path,'train.txt'), 'r') as f:
        train_list = f.readlines()
    with open(os.path.join(path,'test.txt'), 'r') as f:
        test_list = f.readlines()

    train_label = [class_label[img.split('/')[0]] for img in train_list]
    test_label = [class_label[img.split('/')[0]] for img in test_list]
    # random.seed(0)
    # random.shuffle(train_list)
    # train_lists = np.array_split(train_list, n_task)
    TrainSet = TrainDataset(path, train_list, transform, train_label)
    TestSet = TestDataset(path, test_list, transform, test_label)
    return gen_class_il_data(TrainSet, TestSet, n_task, 20, path, transform)


def gen_class_il_data(train_dataset, test_dataset, n_task, n_classes_per_task, path, transform):
    i = 0
    train_datasets = []
    test_datasets = []
    while i < n_task*n_classes_per_task:
        if i + n_classes_per_task >= 100:
            train_mask = np.logical_and(np.array(train_dataset.label) >= i,
                np.array(train_dataset.label) < i + n_classes_per_task + 1)
            test_mask = np.logical_and(np.array(test_dataset.label) >= 0,
                np.array(test_dataset.label) < i + n_classes_per_task + 1)
        else:
            train_mask = np.logical_and(np.array(train_dataset.label) >= i,
                np.array(train_dataset.label) < i + n_classes_per_task)
            test_mask = np.logical_and(np.array(test_dataset.label) >= 0,
                np.array(test_dataset.label) < i + n_classes_per_task)
        train_data = np.array(train_dataset.list)[train_mask]
        train_label = np.array(train_dataset.label)[train_mask]
        test_data = np.array(test_dataset.list)[test_mask]
        test_label = np.array(test_dataset.label)[test_mask]

        train_set = TrainDataset(path, train_data, transform, train_label)
        test_set = TestDataset(path, test_data, transform, test_label)
        train_datasets.append(train_set)
        test_datasets.append(test_set)

        i += n_classes_per_task

    return train_datasets, test_datasets

# transform = None
# train_datasets, test_datasets = get_food101_n_data('../data/food-101', transform, 5)
# print(len(train_datasets[1]))
# print(len(test_datasets[0]))
