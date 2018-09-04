import os
import click
import warnings
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms
from test_augmentation import FlipAndRotateCrop

import feature1_module
import feature2_module


warnings.filterwarnings('ignore')


def get_feature1(task):
    num_classes = task_list[task]
    # resnet
    resnet = feature1_module.model(num_classes)
    resnet_features = nn.Sequential(*list(resnet.children())[:-1]) # 2048
    return resnet_features


def get_feature2(task):
    num_classes = task_list[task]
    net = feature2_module.model(num_classes)
    features = nn.Sequential(*list(net.children())[:-1]) # 1536
    return features


class final_model(nn.Module):
    def __init__(self, block1, block2, num_classes):
        super(final_model, self).__init__()
        self.feature1 = block1
        self.feature2 = block2

        for param in self.feature1.parameters():
            param.requires_grad = False

        for param in self.feature2.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(3584, num_classes)

    def forward(self, x):
        x1 = self.feature1(x)
        x2 = self.feature2(x)
        x1 = x1.view(x1.shape[0], -1)  # (batch_size, 2048)
        x2 = x2.view(x2.shape[0], -1)

        out = torch.cat((x1, x2), 1)
        out = self.dropout(out)
        out = self.last_linear(out)
        return out

@click.command()
@click.option('--task', type=str, required=True)
@click.option('--input_directory', type=str, required=True)
def predict(task, input_directory):
    test_batch = 2
    num_classes = task_list[task]

    block1 = get_feature1(task)
    block2 = get_feature2(task)
    net = final_model(block1, block2, num_classes)

    update_state = torch.load('/root/Project/models/model_2/%s.pth.tar' % task)['state_dict']
    net.load_state_dict(update_state)

    # net = torch.nn.DataParallel(net)
    net.cuda()
    net.eval()

    # data
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    pred_transform = transforms.Compose([
        transforms.Resize(512),
        FlipAndRotateCrop(500, [6, 14, 20]),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
    ])

    dataset_root = input_directory
    sub_root = '/root/Project/src/model_2/result/'

    print('model_2: predicting', task)
    f_out = open(sub_root + '%s.csv' % task, 'w')
    with open(dataset_root + 'Tests/question.csv', 'r') as f_in:
        lines = f_in.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    task_tokens = [t for t in tokens if t[1] == task]
    softmax = nn.Softmax(dim=0)
    for idx, batch in tqdm(enumerate(task_tokens), total=len(task_tokens)):
        with torch.no_grad():
            path, task, _ = batch
            img_path = os.path.join(dataset_root, path)
            img = Image.open(img_path)
            img = pred_transform(img)

            if len(img) % test_batch == 0:
                round_num = int(len(img) / test_batch)
            else:
                round_num = int(len(img) / test_batch) + 1

            out_list = []
            for idx in range(round_num):
                start = idx * test_batch
                end = (idx + 1) * test_batch
                if end > len(img):
                    end = len(img)
                img_ = img[start:end]
                out_0 = net(img_.cuda())
                out_list.append(out_0)
                del img_, out_0
            out_ = torch.cat(out_list, dim=0)
            out_mean = torch.mean(out_, dim=0)

            out = softmax(out_mean)
            out = out.cpu().data.numpy().tolist()
            pred_out = ';'.join(["%.8f" % (o) for o in out])
            line_out = ','.join([path, task, pred_out])
            f_out.write(line_out + '\n')
            del img, out_mean, out

    f_out.close()


task_list = {
    'collar_design_labels': 5,
    'skirt_length_labels': 6,
    'lapel_design_labels': 5,
    'neckline_design_labels': 10,
    'coat_length_labels': 8,
    'neck_design_labels': 5,
    'pant_length_labels': 6,
    'sleeve_length_labels': 9
}
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    predict()
