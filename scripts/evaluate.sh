#!/bin/bash
# evaluate_model.sh: Script to evaluate a trained model.

# example: evaluating the trained model
python src/evaluate.py --data-path path/to/Sketchy --csv-path path/to/PSC6K \
                       --arch resnet18 --checkpoint path/of/model/weights