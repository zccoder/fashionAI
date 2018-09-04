#!/bin/sh

python3.6 /root/Project/src/model_2/predict.py --task coat_length_labels --input_directory "$1"
python3.6 /root/Project/src/model_2/predict.py --task collar_design_labels --input_directory "$1"
python3.6 /root/Project/src/model_2/predict.py --task lapel_design_labels --input_directory "$1"
python3.6 /root/Project/src/model_2/predict.py --task neck_design_labels --input_directory "$1"
python3.6 /root/Project/src/model_2/predict.py --task neckline_design_labels --input_directory "$1"
python3.6 /root/Project/src/model_2/predict.py --task pant_length_labels --input_directory "$1"
python3.6 /root/Project/src/model_2/predict.py --task skirt_length_labels --input_directory "$1"
python3.6 /root/Project/src/model_2/predict.py --task sleeve_length_labels --input_directory "$1"

python3.6 /root/Project/src/model_2/concat_result.py
