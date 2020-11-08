from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from random import random
from PIL import Image
import time
import cv2
import numpy as np
import re
import os

class LabeledDataset(Dataset):
    def __init__(self,path,list,transform,label):
        self.list = list
        self.len = len(list)
        self.path = path
        self.resize = transforms.Resize((32,32))
        self.transform = transform
        self.label = label

    def __getitem__(self,index):
        pic_path = os.path.join(self.path,'images',self.list[index]+'.jpg')
        img = Image.open(pic_path).convert('RGB')
        # img=cv2.resize(np.asarray(img),(64,64),interpolation=cv2.INTER_NEAREST)  #修改图片的尺寸
        img = self.transform(img)
        # img = self.resize(img)
        # print(self.list[index].split('/')[0])
        # print(self.label[self.list[index].split('/')[0]])
        return np.array(img),self.label[self.list[index].split('/')[0]]

    def __len__(self):
        return self.len

class UnlabeledDataset(Dataset):
    def __init__(self,path,list,transform,label):
        self.list = list
        self.len = len(list)
        self.path = path
        self.resize = transforms.Resize((32,32))
        self.transform = transform

    def __getitem__(self,index):
        pic_path = os.path.join(self.path,'images',self.list[index]+'.jpg')
        img = Image.open(pic_path).convert('RGB')
        # img = self.resize(img)
        img = self.transform(img)
        return img,1

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
        return np.array(img),self.label[self.list[index].split('/')[0]]

    def __len__(self):
        return self.len


def get_food101Data(path,LabeledPercent,labeled_tranform,unlabeled_transform):
    i=0
    class_label={}
    with open(os.path.join(path,'classes.txt'), 'r') as f:
        line = f.readline()
        while(line):
            class_label[line.replace("\n","")] = i
            i+=1
            line = f.readline()
    with open(os.path.join(path,'train.txt'), 'r') as f:
        picture_list = f.readlines()
    LabeledList=[]
    UnlabeledList=[]
    for pic in picture_list:
        if(random()>LabeledPercent):
            UnlabeledList.append(pic.replace("\n",""))
        else:
            LabeledList.append(pic.replace("\n",""))
    
    with open(os.path.join(path,'test.txt'), 'r') as f:
        test_list = f.readlines()
    LabeledSet = LabeledDataset(path,LabeledList,labeled_tranform,class_label)
    UnlabeledSet = UnlabeledDataset(path,UnlabeledList,unlabeled_transform,class_label)
    TestSet = TestDataset(path,test_list,labeled_tranform,class_label)
    return LabeledSet,UnlabeledSet,TestSet


# LabeledSet,UnlabeledSet,TestSet = get_food101('./food-101/',0.01)
# labeled_trainloader = DataLoader(
#     LabeledSet,
#     shuffle = True,
#     batch_size=16,
#     num_workers=4,
#     drop_last=True)