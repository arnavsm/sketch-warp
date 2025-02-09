import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import create_conditional_resnet
from warp_estimator import MultiScaleSTN


class CorrespondenceModel(nn.Module):
    """Photo-Sketch Correspondence Model combining a feature encoder and STN."""
    def __init__(self, framework, backbone, dim=128, corr_layers=[3], feat_size=16, stn_sizes=[4, 8, 16],
                 stn_layers=5, pretrained_encoder="", replace_stride_with_dilation=[False, False, False], **kwargs):
        super(CorrespondenceModel, self).__init__()
        self.corr_layers = corr_layers
        self.feat_size = feat_size
        self.stn_sizes = stn_sizes

        # Feature encoders
        self.encoder_q = backbone(num_classes=dim, pretrained=pretrained_encoder,
                                  replace_stride_with_dilation=replace_stride_with_dilation)
        self.encoder_k = backbone(num_classes=dim, pretrained=pretrained_encoder,
                                  replace_stride_with_dilation=replace_stride_with_dilation)
        self.encoder_q.fc = nn.Identity()
        self.encoder_k.fc = nn.Identity()

        # Contrastive learning framework
        self.framework = framework(self.encoder_q, self.encoder_k, dim, **kwargs)

        # Spatial transformer network
        self.stn = MultiScaleSTN(input_size=feat_size, output_sizes=stn_sizes, num_layers=stn_layers, feature_layers=corr_layers)

        # Identity map for dense correspondence
        self.register_buffer("pos_map", F.affine_grid(
            torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0),
            [1, 1, feat_size, feat_size], align_corners=True
        ).permute(0, 3, 1, 2))

    def forward_framework(self, im_q, im_k, cond_q, cond_k):
        """Forward pass through the contrastive learning framework."""
        output, target, res_q, res_k = self.framework(
            im_q=im_q, im_k=im_k, cond_q=cond_q, cond_k=cond_k, return_map=True
        )
        return output, target, res_q, res_k

    def forward_backbone(self, im, cond, corr_only=False):
        """Forward pass through the feature encoder backbone."""
        features, feature_maps = self.framework.encoder_q(
            im, cond, return_map=True, corr_layer=self.corr_layers if corr_only else None
        )
        return features, feature_maps

    def forward_stn(self, map1, map2, dense_mtx=False):
        """Forward pass through the STN."""
        fwd_flow = self.stn(map1, map2)
        bwd_flow = self.stn(map2, map1)
        if dense_mtx:
            dense_corr = self._dense_correspondence([fwd_flow])
            return fwd_flow, bwd_flow, dense_corr
        else:
            return fwd_flow, bwd_flow

    def _dense_correspondence(self, fwd_flows):
        """Compute dense correspondence matrix for visualization."""
        batch_size = fwd_flows[0].size(0)
        pos_map = self.pos_map.repeat(batch_size, 1, 1, 1)
        fwd_map = F.grid_sample(pos_map, fwd_flows[0], align_corners=True, padding_mode="border")
        for flow in fwd_flows[1:]:
            fwd_map = F.grid_sample(fwd_map, flow, align_corners=True, padding_mode="border")
        corr_matrix = torch.cdist(
            pos_map.permute(0, 2, 3, 1).view(batch_size, -1, 2),
            fwd_map.permute(0, 2, 3, 1).view(batch_size, -1, 2)
        )
        return corr_matrix

    def compute_similarity(self, map1, map2):
        """Compute similarity and weight maps for feature layers."""
        map1_list, map2_list = [], []
        for layer in self.corr_layers:
            map1_list.append(F.interpolate(map1[f"layer{layer}"], (self.feat_size, self.feat_size), mode="bilinear"))
            map2_list.append(F.interpolate(map2[f"layer{layer}"], (self.feat_size, self.feat_size), mode="bilinear"))
        map1 = torch.cat(map1_list, dim=1)
        map2 = torch.cat(map2_list, dim=1)

        batch_size, _, width, height = map1.size()
        dim = width * height

        # Flatten and compute normalized correlation
        map1 = map1.view(batch_size, dim, -1).permute(0, 2, 1)
        map2 = map2.view(batch_size, -1, dim)
        corr = torch.matmul(map1, map2).view(batch_size, dim, dim)
        corr = F.normalize(corr, p=1, dim=1)
        corr = F.normalize(corr, p=1, dim=2)

        weight_map = corr.max(dim=2)[0]
        weight_map = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min())
        return [corr], [weight_map.detach()]