#!/bin/bash
# train_model.sh: Script to train a machine learning model.

# example: training feature encoder with ResNet-18 backbone 
python src/train.py --data-path path/to/Sketchy --csv-path path/to/PSC6K \
                    --save-dir path/to/weights/saving --log-dir path/to/logging \
                    --resume-pretrained-encoder path/to/imagenet/pretrained/weights \
                    --task encoder --arch resnet18 \
                    --lr 0.03 --knn-freq 5 -j 16