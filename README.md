# SketchWarp

Self-supervised learning framework for establishing dense correspondences between photographs and sketches, enabling automatic image warping onto sketch templates. Built with PyTorch.

## Features

- Implements the paper [`Learning Dense Correspondences between Photos and Sketches`](misc/2307.12967v1.pdf)
- Self-supervised training and evaluation pipeline
- Image-to-Sketch warping capabilities
- PyTorch-based implementation

## Installation

Create the conda environment:
```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate warp_env
```

## Requirements

Environment requirements are specified in `environment.yml`. Key dependencies include:
- PyTorch >= 2.5
- Python >= 3.10
- CUDA (optional for inference, for GPU support)

## License

This repository is licensed under the `MIT License`, allowing for its use in research and educational purposes.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{sketchwarp2025,  
  title={SketchWarp: Self-Supervised Learning for Photo-Sketch Dense Correspondences},  
  author={[Arnav Samal]},  
  year={2025}  
}  
```