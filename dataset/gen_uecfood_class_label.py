import os
import shutil
import sys
from time import ctime
import cv2

label_path = '../data/UECFOOD100/train.txt'
target_path = '../data/UECFOOD100/train_class_label.txt'

print('script start at: ', ctime())

label_list = []
with open(label_path, 'r') as f:
    l = f.readline()
    while(l):
        label_list.append(int(l.split('/')[1]))
        l = f.readline()

print(label_list)
print('script stop at: ', ctime())