import os
import shutil
import sys
from time import ctime
import cv2

'''
这个脚本是把从网上下载的数据集整理成本项目需要的数据集格式，主要是进行了图像复制和利用boundingbox信息裁切图像。
'''

original_data_path = '/home/hatsunemiku/dev/data/UECFOOD100/'
dataset_path = '../data/UECFOOD100/'

# 是否重新进行这个脚本
if os.path.exists(dataset_path):
    choice = input('do you want to generate the dataset again?[N/y]')
    if choice == 'y':
        shutil.rmtree(dataset_path)
    else:
        sys.exit('\nthe old dataset left!\n')
os.system('mkdir ' + dataset_path)


def parse_bb_info(filename):

    with open(filename) as f:
        f.readline()
        items = [line.split() for line in f.readlines()]

    return items


print('script start at: ', ctime())

categories = [str(i) for i in range(1, 101)]
for ctg in categories:
    path_old = original_data_path + ctg
    path_new = dataset_path + ctg
    os.system('mkdir ' + path_new)

    items = parse_bb_info(path_old + '/' + 'bb_info.txt')
    count = 0
    for img_name, xmin, ymin, xmax, ymax in items:
        img = cv2.imread(path_old + '/' + img_name + '.jpg')
        cv2.imwrite(
            path_new + '/' + img_name + '.jpg',
            img[int(ymin):int(ymax), int(xmin):int(xmax)])

        count += 1

    print('category %4s has %4d images' % (ctg, count))

print('script stop at: ', ctime())