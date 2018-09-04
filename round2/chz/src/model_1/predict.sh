#!/bin/sh

python3.6 /root/Project/src/model_1/predict.py --cloth coat_length_labels --input_directory "$1"
python3.6 /root/Project/src/model_1/predict.py --cloth collar_design_labels --input_directory "$1"
python3.6 /root/Project/src/model_1/predict.py --cloth lapel_design_labels --input_directory "$1"
python3.6 /root/Project/src/model_1/predict.py --cloth neck_design_labels --input_directory "$1"
python3.6 /root/Project/src/model_1/predict.py --cloth neckline_design_labels --input_directory "$1"
python3.6 /root/Project/src/model_1/predict.py --cloth pant_length_labels --input_directory "$1"
python3.6 /root/Project/src/model_1/predict.py --cloth skirt_length_labels --input_directory "$1"
python3.6 /root/Project/src/model_1/predict.py --cloth sleeve_length_labels --input_directory "$1"

python3.6 /root/Project/src/model_1/concat_result.py
