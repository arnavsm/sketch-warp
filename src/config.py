from dataclasses import dataclass


@dataclass
class Config:
    """Configuration: Parameters, Models, etc."""

    input_image_size = 256
    training_epochs = 1300
    estimator_epochs = 1200
    total_epochs = 2500
    learning_rate = {"initial": 0.03, "estimator": 0.003}
    weight_decay = 1e-4
    batch_size = 256
    mixed_precision = True

    data_dir = "data"
    logging_dir = "logs"

    feature_encoder = {
        "architecture": {
            "ResNet-18": "resnet18.a1_in1k",
            "ResNet-101": "resnet101.a1h_in1k",
        },  # we shall try others as well
        "pretrained_weights": "MoCo",
        "embedding_dim": 128,
        "momentum": 0.999,
        "temperature": 0.07,
        "mlp_head_layers": 2,
    }

    memory_queue = 8192

    loss_weights = {"lambda_sim": 0.1, "lambda_con": 1.0}
    loss_computation = {"stages": [2, 3], "temperature": 0.001}

    @staticmethod
    def display():
        """Display the configuration for verification."""
        for key, value in Config.__dict__.items():
            if not key.startswith("__"):
                print(f"{key}: {value}")


# Example usage
Config.display()
