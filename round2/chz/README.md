# 预测逻辑设计与步骤说明
- 本队伍在最终提交模型中没有采用人体检测等预处理。
- 测试图片经过翻转、旋转等测试增强后直接输入模型中进行预测。（predict.py)
- 每个模型的八个任务都会分别预测，预测完之后八个任务的结果拼到一起。（concat_result.py)
- 两个模型的预测结果进行融合即可得到最终结果。(ensemble.py)
# 模型结构
## 模型一
### 结构设计
- 模型一具有三个提取特征的分支，分支一采用senet的结构，分支二采用改进后的inception v4（改进可以认为是将多个浅层特征拼接到最终的feature map当中，分支三采用densenet的结构。
- 三个分支产生的特征不会用独立的全连接层进行预测，而是将特征进行全局池化之后拼成在一起成为一个10400×1的feature map，该feature map接一个全连接层对结构进行预测。
- 模型一是多任务的形式，对于四种长度标签，合并为一个有关长度的四任务模型，对于四种设计标签，合并为一个有关设计的四任务模型，故最终八个任务只有两个模型文件（结构和参数相同，仅最后的全连接层不同）。
### 结构定义文件位置
- 分支一：/root/Project/src/model\_1/feature1_module.py
- 分支二：/root/Project/src/model\_1/feature2_module.py
- 分支三：/root/Project/src/model\_1/feature3_module.py
- 最终模型：/root/Project/src/model\_1/final_model.py
### 模型大小
- 462M
### 参数文件位置
- /root/Project/models/model\_1/multi\_task_length.pth.tar
- /root/Project/models/model\_1/multi\_task_design.pth.tar
## 模型二
### 结构设计
- 模型二具有两个提取特征的分支，分支一采用resnet的结构，分支二采用inception resnet v2的结构。
- 两个分支产生的特征不会用独立的全连接层进行预测，而是将特征进行全局池化之后拼成在一起成为一个3584×1的feature map，该feature map接一个全连接层对结构进行预测。
- 模型二是单任务的形式，故八个任务具有八个模型文件（结构和参数相同，仅最后的全连接层不同）。
### 结构定义文件位置
- 分支一：/root/Project/src/model\_2/feature1_module.py
- 分支二：/root/Project/src/model\_2/feature2_module.py
- 最终模型：/root/Project/src/model\_2/predict.py
### 模型大小
- 451M
### 参数文件位置
- /root/Project/models/model\_2/coat\_length_labels.pth.tar
- /root/Project/models/model\_2/collar\_design_labels.pth.tar
- /root/Project/models/model\_2/lapel\_design_labels.pth.tar
- /root/Project/models/model\_2/neck\_design_labels.pth.tar
- /root/Project/models/model\_2/neckline\_design_labels.pth.tar
- /root/Project/models/model\_2/pant\_length_labels.pth.tar
- /root/Project/models/model\_2/skirt\_length_labels.pth.tar
- /root/Project/models/model\_2/sleeve\_length_labels.pth.tar
# 输出结果说明
- /root/Project/src/model\_1/result——将产生九个结果，其中八个分别为八个任务各自的预测结果，最后将这八个结果通过上一级文件夹内的concat_result.py拼接到一起，为第九个结果
- /root/Project/src/model\_1/result——同上