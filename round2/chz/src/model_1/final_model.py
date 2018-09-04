import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import feature1_module
import feature2_module
import feature3_module

def final_model(task):
    if task == 'multi_task_design':
        feature1 = feature1_module.multi_task_design().cuda()
        feature2 = feature2_module.multi_task_design().cuda()
        feature3 = feature3_module.multi_task_design().cuda()
    else:
        feature1 = feature1_module.multi_task_length().cuda()
        feature2 = feature2_module.multi_task_length().cuda()
        feature3 = feature3_module.multi_task_length().cuda()

    feature1 = nn.Sequential(*list(feature1.children())[:-4])
    feature2 = nn.Sequential(*list(feature2.children())[:-4])
    feature3 = nn.Sequential(*list(feature3.children())[:-4])

    model = FinalModel(feature1, feature2, feature3, task)
    return model

class FinalModel(nn.Module):
    def __init__(self, feature1, feature2, feature3, task):
        super(FinalModel, self).__init__()
        self.feature1 = feature1
        self.feature2_1 = feature2[0]
        self.feature2_2 = feature2[1]
        self.feature2_3 = feature2[2]
        self.feature2_4 = feature2[3]
        self.feature2_5 = feature2[4]
        self.feature3 = feature3

        self.drop_out = nn.Dropout()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if task == 'multi_task_design':
            self.fc4_1 = nn.Linear(10400, 5)  # collar
            self.fc4_2 = nn.Linear(10400, 5)  # lapel
            self.fc4_3 = nn.Linear(10400, 5)  # neck
            self.fc4_4 = nn.Linear(10400, 10)  # neckline

        else:
            self.fc4_1 = nn.Linear(10400, 8)  # coat
            self.fc4_2 = nn.Linear(10400, 6)  # pant
            self.fc4_3 = nn.Linear(10400, 6)  # skirt
            self.fc4_4 = nn.Linear(10400, 9)  # sleeve

    def forward(self, x):
        x1 = self.feature1(x)
        x2_0 = self.feature2_1(x)
        x2_1 = self.feature2_2(x2_0)
        x2_2 = self.feature2_3(x2_1)
        x2_3 = self.feature2_4(x2_2)
        x2_4 = self.feature2_5(x2_3)
        x2_0, x2_1, x2_2, x2_3, x2_4 = self.avg_pool(x2_0), self.avg_pool(x2_1), self.avg_pool(x2_2), self.avg_pool(x2_3), self.avg_pool(x2_4)
        x2 = torch.cat((x2_0, x2_1, x2_2, x2_3, x2_4), 1)
        x3 = self.feature3(x)
        x3 = F.relu(x3, inplace=True)
        x1, x2, x3 = x1.view(x1.shape[0], -1), x2.view(x2.shape[0], -1), x3.view(x3.shape[0], -1)
        
        x4 = torch.cat((x1, x2, x3), 1)
        x4 = self.drop_out(x4)

        out1, out2, out3, out4 = self.fc4_1(x4), self.fc4_2(x4), self.fc4_3(x4), self.fc4_4(x4)

        return out1, out2, out3, out4
