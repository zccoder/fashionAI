#!/bin/bash 

# Train_data ...
echo 'Start training ...'
data_dir_list=(
/home2/data/fashionAI/train_data/Images/coat_length_labels 
)

num=${#data_dir_list[@]}
#echo $num
infer_data_list=(
img_list/train_coat.txt
)

model_dir="./model"

output_dir_list=(
/home2/data/fashionAI/Deeplabv3_train/Images/coat_length_labels
)

for((i=0;i<num;i++)){
    echo ${data_dir_list[i]};
    python inference.py --data_dir ${data_dir_list[i]} --infer_data_list ${infer_data_list[i]} --model_dir $model_dir --output_dir ${output_dir_list[i]}
}
