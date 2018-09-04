import os
import click
from final_model import final_model

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict, defaultdict
from test_augmentation import FlipAndRotateCrop

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

@click.command()
@click.option('--cloth', type=str, required=True)
@click.option('--input_directory', type=str, required=True)
def main(cloth, input_directory='/data/Attributes/Round2b/'):
    def default_loader(path):
        return Image.open(path).convert('RGB')

    class TestDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['real_path'], row['path']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, path = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, path

        def __len__(self):
            return len(self.imgs)

    # transfer parameters
    def load_pre_cloth_model_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def test(test_loader, model, cloth):
        csv_map = OrderedDict({'ImageName': [], 'AttrKey': [], 'AttrValueProbs': []})
        # switch to evaluate mode
        model.eval()
        smax = nn.Softmax()
        for i, (images, filepath) in enumerate(tqdm(test_loader)):
            bs, ncrops, c, h, w = images.size()
            image_var = torch.tensor(images, requires_grad=False).cuda(async=True)

            with torch.no_grad():
                y_pred = model(image_var.view(-1, c, h, w))[clothes_types.index(cloth)]  # fuse batch size and ncrops
                y_pred = y_pred.view(bs, ncrops, -1).mean(1)  # avg over crops
                # get the index of the max log-probability
                smax_out = smax(y_pred)[0]
            prob = ';'.join('%.8f' % output for output in smax_out.data)
            csv_map['ImageName'].append(filepath[0])
            csv_map['AttrKey'].append(cloth)
            csv_map['AttrValueProbs'].append(prob)

        result = pd.DataFrame(csv_map)
        result.to_csv('/root/Project/src/model_1/result/%s.csv' % cloth, index=False)
        return

    print('model_1: predicting', cloth)
    if 'design' in cloth:
        task = 'multi_task_design'
        clothes_types = ['collar_design_labels', 'lapel_design_labels', 'neck_design_labels', 'neckline_design_labels']
    else:
        task = 'multi_task_length'
        clothes_types = ['coat_length_labels', 'pant_length_labels', 'skirt_length_labels', 'sleeve_length_labels']

    # build a model
    model = final_model(task)
    model = torch.nn.DataParallel(model).cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([
                                  transforms.Resize((512, 512)),
                                  FlipAndRotateCrop(500, [5, 13, 21]),
                                  transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                  transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
                                  ])

    test_data_list = pd.read_csv(input_directory + 'Tests/question.csv', header=None)
    test_data_list.columns = ['path', 'type', 'label']
    test_data_list['real_path'] = input_directory + test_data_list['path']
    test_data_list = test_data_list[test_data_list['type'] == cloth]
    
    test_data = TestDataset(test_data_list, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=False, num_workers=20)

    best_model = torch.load('/root/Project/models/model_1/%s.pth.tar' % task)
    load_pre_cloth_model_dict(model, best_model['state_dict'])
    test(test_loader=test_loader, model=model, cloth=cloth)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
