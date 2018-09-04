# coding=utf-8
import os

# 生成所需要的img_list数据

data_path = '/home2/data/fashionAI/test_data/Images/neck_design_labels'
out_name = 'neck_img_list_test.txt'

img_list = [im for im in os.listdir(data_path) if im.endswith('.jpg')]
out = open(out_name, 'w')
for img_path in img_list:
	out.writelines(img_path + '\n')
