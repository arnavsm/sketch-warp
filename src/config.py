from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Data paths
    csv_path: str = "data/annotations_info"
    data_path: str = "data"

    # Job control
    workers: int = 32
    epochs: int = 1500
    start_epoch: int = 0
    warmup_epochs: int = 100
    batch_size: int = 256
    optim: str = 'sgd'
    lr: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 1e-4
    print_freq: int = 1
    knn_freq: int = 50
    pck_freq: int = 100
    plot_freq: int = 50
    save_freq: int = 50
    log_dir: str = "logs"
    save_dir: str = "models"
    resume: str = ""
    resume_encoder: str = ""
    resume_pretrained_encoder: str = ""
    world_size: int = 1
    rank: int = 0
    dist_url: str = "tcp://localhost:12355"
    dist_backend: str = "nccl"
    seed: int = 0
    gpu: int = None
    multiprocessing_distributed: bool = True
    writer_name: str = "warp"
    comment: str = ""

    # MoCo parameters
    moco_dim: int = 128
    moco_k: int = 8192
    moco_m: float = 0.999
    moco_t: float = 0.07
    corr_t: float = 0.001

    # Loss weights
    clr_loss_weight: float = 1.0
    sim_loss_weight: float = 0.1
    con_loss_weight: float = 1.0
    syn_loss_weight: float = 0.0

    # Training details
    task: str = "encoder"  # 'encoder', 'estimator', 'both'
    arch: str = "resnet18"  # 'resnet18', 'resnet50', 'resnet101'
    supervision: str = "pair"  # 'instance', 'pair', 'class'
    layer: List[int] = field(default_factory=lambda: [2, 3])
    trans_type: List[str] = field(default_factory=lambda: ['affine', 'tps', 'afftps'])
    feat_size: int = 16
    stn_size: int = 16
    stn_layer: int = 5
    break_training: int = None

    # Ablation
    cbn: bool = True
    cos: bool = True
    weighted: bool = True
    freeze: bool = True
    perceptual: bool = True
