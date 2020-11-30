from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from random import random
from PIL import Image
import numpy as np
import os

class LabeledDataset(Dataset):
    def __init__(self,path,list,transform):
        self.list = list
        self.len = len(list)
        self.path = path
        self.transform = transform

    def __getitem__(self,index):
        pic_path = os.path.join(self.path, self.list[index].replace("\n",""))
        img = Image.open(pic_path).convert('RGB')
        # img=cv2.resize(np.asarray(img),(64,64),interpolation=cv2.INTER_NEAREST)  #修改图片的尺寸
        img = self.transform(img)
        # img = self.resize(img)
        # print(self.list[index].split('/')[0])
        # print(self.label[self.list[index].split('/')[0]])
        return np.array(img), int(self.list[index].split('/')[1])-1

    def __len__(self):
        return self.len

class UnlabeledDataset(Dataset):
    def __init__(self,path,list,transform):
        self.list = list
        self.len = len(list)
        self.path = path
        self.transform = transform

    def __getitem__(self,index):
        pic_path = os.path.join(self.path, self.list[index].replace("\n",""))
        img = Image.open(pic_path).convert('RGB')
        img = self.transform(img)
        return img,1

    def __len__(self):
        return self.len

class TestDataset(Dataset):
    def __init__(self,path,list,transform):
        self.list = list
        self.len = len(list)
        self.path = path
        self.transform = transform

    def __getitem__(self,index):
        pic_path = os.path.join(self.path, self.list[index].replace("\n",""))
        img = Image.open(pic_path).convert('RGB')
        img = self.transform(img)
        return np.array(img), int(self.list[index].split('/')[1])-1

    def __len__(self):
        return self.len


def get_uecfood_data(path,labeled_percent,labeled_tranform,unlabeled_transform):
    
    with open(os.path.join(path,'test.txt'), 'r') as f:
        test_list = f.readlines()

    filename = 'train_label_{}.txt'.format(labeled_percent)
    with open(os.path.join(path, filename), 'r') as f:
        train_label_list = f.readlines()
    
    filename = 'train_unlabel_{}.txt'.format(labeled_percent)

    with open(os.path.join(path, filename), 'r') as f:
        train_unlabel_list = f.readlines()
        
    path = path.rsplit('/', 1)[0]
    LabeledSet = LabeledDataset(path,train_label_list,labeled_tranform)
    UnlabeledSet = UnlabeledDataset(path,train_unlabel_list,unlabeled_transform)
    TestSet = TestDataset(path,test_list,labeled_tranform)
    return LabeledSet,UnlabeledSet,TestSet