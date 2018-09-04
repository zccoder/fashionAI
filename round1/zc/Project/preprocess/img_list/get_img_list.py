# coding=utf-8
import os

# 生成所需要的img_list数据
# ========= 针对train =========
train_path = '/home2/data/fashionAI/train_data/Images/'
# ========= 针对test ========
test_path = '/home2/data/fashionAI/test_data/Images'

data_path_list = [
    'coat_length_labels',
    'neckline_design_labels',
    'collar_design_labels',
    'pant_length_labels',
    'lapel_design_labels',
    'skirt_length_labels',
    'neck_design_labels',
    'sleeve_length_labels'
]

out_list = [
    'coat.txt',
    'neckline.txt',
    'collar.txt',
    'pant.txt',
    'lapel.txt',
    'skirt.txt',
    'neck.txt',
    'sleeve.txt'
]

# ========== 开始生成数据 =============
print('Getting train data ...')
for i in range(0, len(data_path_list)):
   	data_path = os.path.join(train_path, data_path_list[i])
        out_name = 'train_' + out_list[i]

	img_list = [im for im in os.listdir(data_path) if im.endswith('.jpg')]       
	out = open(out_name, 'w')
	for img_path in img_list:
		out.writelines(img_path + '\n')


print('Getting test data ...')
for i in range(0, len(data_path_list)):
   	data_path = os.path.join(test_path, data_path_list[i])
        out_name = 'test_' + out_list[i]

	img_list = [im for im in os.listdir(data_path) if im.endswith('.jpg')]       
	out = open(out_name, 'w')
	for img_path in img_list:
		out.writelines(img_path + '\n')
