import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms


def predict(task):
    # model
    saved_model = 'checkpoint/' + task + '_model.pkl'
    net = torch.load(saved_model)
    net.cuda()
    # data
    pred_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(500),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])

    dataset_root = '/home/hsun/coco-sun/siat-mmlab-data/FashionAI_Data/classification/'
    sub_root = 'subs_b/'

    logging.info('Starting Prediction for %s \n' % task)
    f_out = open(sub_root + '%s.csv' % (task), 'w')
    with open(dataset_root + 'z_rank/Tests/question.csv', 'r') as f_in:
        lines = f_in.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    task_tokens = [t for t in tokens if t[1] == task]
    softmax = nn.Softmax(dim=1)
    for idx, batch in tqdm(enumerate(task_tokens), total=len(task_tokens)):
        path, task, _ = batch
        img_path = os.path.join(dataset_root + 'z_rank', path)
        img = Image.open(img_path)
        img = pred_transform(img)
        img = img.unsqueeze_(0)
        out = net(Variable(img.cuda()))
        out = softmax(out)
        out = np.squeeze(out.cpu().data.numpy()).tolist()
        pred_out = ';'.join(["%.8f" % (o) for o in out])
        line_out = ','.join([path, task, pred_out])
        f_out.write(line_out + '\n')
    f_out.close()
    print('Submission for %s is done!!!' % task)

if __name__ == '__main__':
    task_list = ['collar_design_labels', 'skirt_length_labels', 'lapel_design_labels',
                 'neckline_design_labels', 'coat_length_labels', 'neck_design_labels',
                 'pant_length_labels', 'sleeve_length_labels']

    for task in task_list:
        predict(task)