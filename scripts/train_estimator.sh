# example: training warp estimator with default parameters
python src/train.py --data-path path/to/Sketchy --csv-path path/to/PSC6K \
                    --save-dir path/to/weights/saving --log-dir path/to/logging \
                    --resume-encoder path/of/model/weights \
                    --task estimator --arch resnet18 \
                    --lr 0.003 --pck-freq 5 \
                    --sim-loss 0.1 --con-loss 1.0 -j 16