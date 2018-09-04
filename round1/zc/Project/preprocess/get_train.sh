#!/bin/bash 

# Train_data ...
echo 'Start training ...'
data_dir_list=(
/home2/data/fashionAI/train_data/Images/coat_length_labels 
/home2/data/fashionAI/train_data/Images/collar_design_labels 
/home2/data/fashionAI/train_data/Images/lapel_design_labels 
/home2/data/fashionAI/train_data/Images/neck_design_labels 
/home2/data/fashionAI/train_data/Images/neckline_design_labels 
/home2/data/fashionAI/train_data/Images/pant_length_labels 
/home2/data/fashionAI/train_data/Images/skirt_length_labels 
/home2/data/fashionAI/train_data/Images/sleeve_length_labels
)

num=${#data_dir_list[@]}
#echo $num
infer_data_list=(
img_list/train_coat.txt
img_list/train_collar.txt
img_list/train_lapel.txt
img_list/train_neck.txt
img_list/train_neckline.txt
img_list/train_pant.txt
img_list/train_skirt.txt
img_list/train_sleeve.txt
)

model_dir="./model"

output_dir_list=(
/home2/data/fashionAI/Deeplabv3_train/Images/coat_length_labels
/home2/data/fashionAI/Deeplabv3_train/Images/collar_design_labels
/home2/data/fashionAI/Deeplabv3_train/Images/lapel_design_labels
/home2/data/fashionAI/Deeplabv3_train/Images/neck_design_labels
/home2/data/fashionAI/Deeplabv3_train/Images/neckline_design_labels
/home2/data/fashionAI/Deeplabv3_train/Images/pant_length_labels
/home2/data/fashionAI/Deeplabv3_train/Images/skirt_length_labels
/home2/data/fashionAI/Deeplabv3_train/Images/sleeve_length_labels
)

for((i=0;i<num;i++)){
    echo ${data_dir_list[i]};
    python inference.py --data_dir ${data_dir_list[i]} --infer_data_list ${infer_data_list[i]} --model_dir $model_dir --output_dir ${output_dir_list[i]}
}
