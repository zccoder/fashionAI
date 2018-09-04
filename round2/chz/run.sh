#!/bin/sh

/root/Project/src/model_1/predict.sh "$1"
/root/Project/src/model_2/predict.sh "$1"
python3.6 /root/Project/src/ensemble.py --output_csv "$2"
