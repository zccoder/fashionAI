#!/bin/bash 

# Train_data ...
echo 'Start testing ...'
data_dir_list=(
/home2/data/fashionAI/z_rank/Images/coat_length_labels 
/home2/data/fashionAI/z_rank/Images/collar_design_labels 
/home2/data/fashionAI/z_rank/Images/lapel_design_labels 
/home2/data/fashionAI/z_rank/Images/neck_design_labels 
/home2/data/fashionAI/z_rank/Images/neckline_design_labels 
/home2/data/fashionAI/z_rank/Images/pant_length_labels 
/home2/data/fashionAI/z_rank/Images/skirt_length_labels 
/home2/data/fashionAI/z_rank/Images/sleeve_length_labels
)

num=${#data_dir_list[@]}
#echo $num
infer_data_list=(
img_list/test_b_coat.txt
img_list/test_b_collar.txt
img_list/test_b_lapel.txt
img_list/test_b_neck.txt
img_list/test_b_neckline.txt
img_list/test_b_pant.txt
img_list/test_b_skirt.txt
img_list/test_b_sleeve.txt
)

model_dir="./model"

output_dir_list=(
/home2/data/fashionAI/Deeplabv3_test_b/Images/coat_length_labels
/home2/data/fashionAI/Deeplabv3_test_b/Images/collar_design_labels
/home2/data/fashionAI/Deeplabv3_test_b/Images/lapel_design_labels
/home2/data/fashionAI/Deeplabv3_test_b/Images/neck_design_labels
/home2/data/fashionAI/Deeplabv3_test_b/Images/neckline_design_labels
/home2/data/fashionAI/Deeplabv3_test_b/Images/pant_length_labels
/home2/data/fashionAI/Deeplabv3_test_b/Images/skirt_length_labels
/home2/data/fashionAI/Deeplabv3_test_b/Images/sleeve_length_labels
)

for((i=0;i<num;i++)){
    echo ${data_dir_list[i]};
    python inference.py --data_dir ${data_dir_list[i]} --infer_data_list ${infer_data_list[i]} --model_dir $model_dir --output_dir ${output_dir_list[i]}
}
