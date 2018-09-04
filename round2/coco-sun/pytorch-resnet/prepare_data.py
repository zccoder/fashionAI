import mxnet
from mxnet import gluon, image
import os, shutil, random

# Rebuild dataset structure
dataset_root = '/home/hsun/coco-sun/siat-mmlab-data/FashionAI_Data/classification/'

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

new_data_path = dataset_root + 'data/train_valid/'
mkdir_if_not_exist([new_data_path])


def create_new_dataset_struct(task, dataset_root, new_data_path):
    # 热身数据与训练数据的图片标记文件
    warmup_label_dir = dataset_root + '/web/Annotations/skirt_length_labels.csv'
    base_label_dir = dataset_root + '/base/Annotations/label.csv'

    image_path = []

    # get images path
    ## for warmup data
    with open(warmup_label_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, tk, label in tokens:
            if tk == task:
                image_path.append((dataset_root + 'web/' + path, label))
    ## for train data
    with open(base_label_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, tk, label in tokens:
            if tk == task:
                image_path.append((dataset_root + 'base/' + path, label))

    # create file path
    mkdir_if_not_exist([new_data_path, task])
    mkdir_if_not_exist([new_data_path, task, 'train'])
    mkdir_if_not_exist([new_data_path, task, 'val'])
    m = len(list(image_path[0][1]))
    for mm in range(m):
        mkdir_if_not_exist([new_data_path, task, 'train', str(mm)])
        mkdir_if_not_exist([new_data_path, task, 'val', str(mm)])

    # move images to relative file path
    n = len(image_path)
    random.seed(1024)
    random.shuffle(image_path)
    train_count = 0
    for path, label in image_path:
        label_index = list(label).index('y')
        if train_count < n * 0.9:
            shutil.copy(path,
                        os.path.join(new_data_path, task, 'train', str(label_index)))
        else:
            shutil.copy(path,
                        os.path.join(new_data_path, task, 'val', str(label_index)))
        train_count += 1

    return image_path

## prepare all tasks
task_list = ['skirt_length_labels', 'coat_length_labels', 'collar_design_labels', 'lapel_design_labels',
            'neck_design_labels', 'neckline_design_labels', 'pant_length_labels', 'sleeve_length_labels']

for task in task_list:
    image_path = create_new_dataset_struct(task, dataset_root, new_data_path)