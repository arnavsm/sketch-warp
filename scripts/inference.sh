#!/bin/bash
# inference.sh: Script to run inference and save visualizations.

# example: running inference with the trained model
python src/inference.py --data-path path/to/Sketchy --csv-path path/to/PSC6K \
                        --arch resnet18 --checkpoint path/of/model/weights \
                        --output-dir path/to/save/visualizations
