import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_identity_map(size):
    """
    Generate an identity displacement field.
    """
    grid = F.affine_grid(
        torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0),
        [1, 1, size, size],
        align_corners=True
    )
    return grid


def build_conv_block(in_channels, out_channels, kernel_size=3, stride=1, activation=True):
    """
    Build a convolutional block with optional activation.
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if activation:
        layers.append(nn.LeakyReLU(0.1))
    return nn.Sequential(*layers)


class MultiScaleSTN(nn.Module):
    def __init__(self, input_size, output_sizes, num_layers, feature_layers):
        """
        Multi-Scale Spatial Transformer Network.
        """
        super().__init__()
        self.feature_layers = feature_layers
        self.input_size = input_size

        self.stn_blocks = nn.ModuleList([
            self._build_stn_block(input_size, size, num_layers) for size in output_sizes
        ])

        self.register_buffer("identity_map", generate_identity_map(input_size))

    def _build_stn_block(self, input_size, output_size, num_layers):
        """
        Build a spatial transformer block for a specific scale.
        """
        downscale_factor = input_size // output_size
        assert downscale_factor in [1, 2, 4, 8], "Invalid downscale factor"

        layers = [
            nn.Conv2d(input_size ** 2 * len(self.feature_layers), input_size ** 2, kernel_size=1, stride=downscale_factor),
            nn.BatchNorm2d(input_size ** 2),
            nn.LeakyReLU(0.1)
        ]

        for _ in range(num_layers):
            layers.append(build_conv_block(input_size ** 2, input_size ** 2 // 2))
            input_size //= 2

        layers += [
            nn.Conv2d(input_size, 2, kernel_size=1),
            nn.Upsample(size=(output_size, output_size), mode="bilinear", align_corners=True)
        ]

        return nn.Sequential(*layers)

    def compute_correlation(self, features1, features2):
        """
        Compute the correlation matrix for given feature maps.
        """
        corrs = []
        for layer in self.feature_layers:
            f1 = features1[f"layer{layer}"]
            f2 = features2[f"layer{layer}"]

            # Flatten and compute correlation
            b, c, h, w = f1.size()
            f1 = f1.view(b, c, h * w).permute(0, 2, 1)
            f2 = f2.view(b, c, h * w)
            corr = torch.matmul(f1, f2).view(b, h * w, h, w)
            corrs.append(corr)

        return torch.cat(corrs, dim=1)

    def forward(self, source_features, target_features, training=False):
        """
        Forward pass through the STN.
        """
        flows = []
        correlation_map = self.compute_correlation(target_features, source_features)

        flow = self.stn_blocks[0](correlation_map).permute(0, 2, 3, 1) + self.identity_map
        flows.append(flow)

        for i, stn_block in enumerate(self.stn_blocks[1:], start=1):
            warped_features = self._warp_features(source_features, flow)
            correlation_map = self.compute_correlation(target_features, warped_features)
            residual_flow = stn_block(correlation_map).permute(0, 2, 3, 1)
            flow = flow + residual_flow
            flows.append(flow)

        if training:
            return flows  # Return flows at all scales during training
        return flows[-1]

    def _warp_features(self, features, flow):
        """
        Warp feature maps using the displacement field (flow).
        """
        warped = {}
        for layer in self.feature_layers:
            warped[f"layer{layer}"] = F.grid_sample(
                features[f"layer{layer}"], flow, mode="bilinear", align_corners=True, padding_mode="zeros"
            )
        return warped