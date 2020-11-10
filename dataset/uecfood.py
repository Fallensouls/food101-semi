from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from random import random
from PIL import Image
import numpy as np
import os

class LabeledDataset(Dataset):
    def __init__(self,path,list,transform,label):
        self.list = list
        self.len = len(list)
        self.path = path
        self.transform = transform
        self.label = label

    def __getitem__(self,index):
        pic_path = os.path.join(self.path, self.list[index])
        img = Image.open(pic_path).convert('RGB')
        # img=cv2.resize(np.asarray(img),(64,64),interpolation=cv2.INTER_NEAREST)  #修改图片的尺寸
        img = self.transform(img)
        # img = self.resize(img)
        # print(self.list[index].split('/')[0])
        # print(self.label[self.list[index].split('/')[0]])
        return np.array(img), self.label[index]-1

    def __len__(self):
        return self.len

class UnlabeledDataset(Dataset):
    def __init__(self,path,list,transform):
        self.list = list
        self.len = len(list)
        self.path = path
        self.transform = transform

    def __getitem__(self,index):
        pic_path = os.path.join(self.path, self.list[index])
        img = Image.open(pic_path).convert('RGB')
        img = self.transform(img)
        return img,1

    def __len__(self):
        return self.len

class TestDataset(Dataset):
    def __init__(self,path,list,transform,label):
        self.list = list
        self.len = len(list)
        self.path = path
        self.transform = transform
        self.label = label

    def __getitem__(self,index):
        pic_path = os.path.join(self.path, self.list[index].replace("\n",""))
        img = Image.open(pic_path).convert('RGB')
        img = self.transform(img)
        return np.array(img),self.label[index]-1

    def __len__(self):
        return self.len


def get_uecfood_data(path,LabeledPercent,labeled_tranform,unlabeled_transform):
    test_label_list = []

    with open(os.path.join(path,'test.txt'), 'r') as f:
        l = f.readline()
        while(l):
            test_label_list.append(int(l.split('/')[1]))
            l = f.readline()

    with open(os.path.join(path,'train.txt'), 'r') as f:
        picture_list = f.readlines()
    LabeledList=[]
    train_label_list = []
    UnlabeledList=[]
    for pic in picture_list:
        if(random()>LabeledPercent):
            UnlabeledList.append(pic.replace("\n",""))
        else:
            train_label_list.append(int(pic.split('/')[1]))
            LabeledList.append(pic.replace("\n",""))
    
    with open(os.path.join(path,'test.txt'), 'r') as f:
        test_list = f.readlines()
    path = path.rsplit('/', 1)[0]
    LabeledSet = LabeledDataset(path,LabeledList,labeled_tranform,train_label_list)
    UnlabeledSet = UnlabeledDataset(path,UnlabeledList,unlabeled_transform)
    TestSet = TestDataset(path,test_list,labeled_tranform,test_label_list)
    return LabeledSet,UnlabeledSet,TestSet