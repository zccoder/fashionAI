#!/bin/bash 

# Train_data ...
echo 'Start testing ...'
data_dir_list=(
/home2/data/fashionAI/test_data/Images/coat_length_labels 
/home2/data/fashionAI/test_data/Images/collar_design_labels 
/home2/data/fashionAI/test_data/Images/lapel_design_labels 
/home2/data/fashionAI/test_data/Images/neck_design_labels 
/home2/data/fashionAI/test_data/Images/neckline_design_labels 
/home2/data/fashionAI/test_data/Images/pant_length_labels 
/home2/data/fashionAI/test_data/Images/skirt_length_labels 
/home2/data/fashionAI/test_data/Images/sleeve_length_labels
)

num=${#data_dir_list[@]}
#echo $num
infer_data_list=(
img_list/test_coat.txt
img_list/test_collar.txt
img_list/test_lapel.txt
img_list/test_neck.txt
img_list/test_neckline.txt
img_list/test_pant.txt
img_list/test_skirt.txt
img_list/test_sleeve.txt
)

model_dir="./model"

output_dir_list=(
/home2/data/fashionAI/Deeplabv3_test/Images/coat_length_labels
/home2/data/fashionAI/Deeplabv3_test/Images/collar_design_labels
/home2/data/fashionAI/Deeplabv3_test/Images/lapel_design_labels
/home2/data/fashionAI/Deeplabv3_test/Images/neck_design_labels
/home2/data/fashionAI/Deeplabv3_test/Images/neckline_design_labels
/home2/data/fashionAI/Deeplabv3_test/Images/pant_length_labels
/home2/data/fashionAI/Deeplabv3_test/Images/skirt_length_labels
/home2/data/fashionAI/Deeplabv3_test/Images/sleeve_length_labels
)

for((i=0;i<num;i++)){
    echo ${data_dir_list[i]};
    python inference.py --data_dir ${data_dir_list[i]} --infer_data_list ${infer_data_list[i]} --model_dir $model_dir --output_dir ${output_dir_list[i]}
}
