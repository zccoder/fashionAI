# Project
## TRAIN
### zc_densenet_bs5.py: 使用预训练的densent201，batch_size=5的模型分别对八个任务进行训练
### zc_Nasnet_width399_danka.py：使用预训练的Nasnet来训练图片尺寸为399，使用单卡对coat类单独训练
### zc_Nasnet_width399.py：使用预训练的Nasnet来训练图片尺寸为399，使用多卡分别对八个任务进行训练
### zc_Nasnet_width468.py：使用预训练的Nasnet来训练图片尺寸为468，使用多卡分别对八个任务进行训练
run.sh： 运行zc_densenet_bs5.py和zc_Nasnet_width399_danka.py运行 . run.sh
#### 其中CUDA_VISIBLE_DEVICES=0(表示采用的显卡)， python zc_densenet_bs5.py(表示运行的python文件) 0 (表示运行的任务，分别为0-7)
run1.sh： 运行zc_Nasnet_width399.py和zc_Nasnet_width468.py运行 . run1.sh
#### 其中python zc_Nasnet_width399.py(表示运行的python文件) 0 (表示运行的任务，分别为0-7)

## TEST
### run_densenet_bs5.py: 载入zc_densenet_bs5.py训练好的权重，进行预测，最后生成zc_0419_densenet_344_0.9773
### run_Nasnet_width399_danka.py：载入zc_Nasnet_width399_danka.py训练好的权重，预测coat任务, 替换zc_0420_nasnet_468_0.9795中的coat，生成zc_0420_nasnet_9795_468_updated_coat_0.98
### run_Nasnet_width399.py：载入zc_Nasnet_width399.py训练好的权重，预测八个任务,生成zc_0420_nasnet_Width399_9773(b_bang)
### run_Nasnet_width468.py：载入zc_Nasnet_width468.py训练好的权重，预测八个任务,生成zc_0420_nasnet_468_0.9795
run.sh： 运行run_densenet_bs5.py和run_Nasnet_width399_danka.py运行 . run.sh
#### 其中CUDA_VISIBLE_DEVICES=0(表示采用的显卡)， python run_densenet_bs5.py(表示运行的python文件) 0 (表示运行的任务，分别为0-7)
run1.sh： 运行run_Nasnet_width399.py和run_Nasnet_width468.py运行 . run1.sh
#### 其中python run_Nasnet_width39.py(表示运行的python文件) 0 (表示运行的任务，分别为0-7)

## PROPROCESS
该文件夹主要是对图片进行分割操作，基于deeplabv3+的github开源项目https://github.com/tensorflow/models/tree/master/research/deeplab进行处理的。其中新增的几个处理文件如下：
### inference.py：载入vgg16预训练的模型，基于人体对衣服进行分割，并保存分割后的图片
### img_list文件夹
#### get_img_list.py: 对初赛A榜的数据集获取图片数据集列表
#### get_test_b_list.py： 对初赛B榜的数据集获取图片数据集列表
#### 各种txt文件：运行get_img_list.py和get_test_b_list.py所生成的文件
### run_all.sh：运行该脚本可自对初赛A榜的所有数据进行分割处理，其中调用到的get_train.sh是对初赛A榜的训练集进行处理, get_test.sh是对初赛A榜的测试集进行处理
#### get_test_b.sh： 对初赛b榜的数据集进行处理