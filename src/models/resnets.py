import torch
import torch.nn as nn
import timm
from timm.models.resnet import ResNet as TimmResNet
from timm.models.registry import register_model
from typing import Optional, Tuple


class CondBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_conds=2):
        super().__init__()
        self.norms = nn.ModuleList([nn.BatchNorm2d(num_features)] * num_conds)

    def forward(self, x, cond):
        return self.norms[cond](x)


class CustomResNet(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=1000):
        super().__init__()
        # Initialize the timm model
        self.base = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        
        # Replace all BatchNorm2d layers with CondBatchNorm2d
        self._replace_bn_layers()
        
        # Store intermediate feature maps
        self.feature_maps = {}
        
    def _replace_bn_layers(self):
        """Replace all BatchNorm2d layers with CondBatchNorm2d"""
        def replace_bn(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, CondBatchNorm2d(child.num_features))
                else:
                    replace_bn(child)
                    
        replace_bn(self.base)
    
    def _forward_impl(self, x: torch.Tensor, cond: int, return_map: bool = True, corr_layer=None):
        y = {}
        
        # Initial layers
        x = self.base.conv1(x)
        x = self.base.bn1(x, cond)
        x = self.base.act1(x)
        x = self.base.maxpool(x)
        
        # Layer1-4 with feature map collection
        x = self.base.layer1(x)
        y["layer1"] = x
        
        x = self.base.layer2(x)
        y["layer2"] = x
        
        x = self.base.layer3(x)
        y["layer3"] = x
        
        if corr_layer is None:
            x = self.base.layer4(x)
            y["layer4"] = x
        else:
            if 4 in corr_layer:
                x = self.base.layer4(x)
                y["layer4"] = x
                
        if corr_layer is None:
            x = self.base.global_pool(x)
            x = x.flatten(1)
            x = self.base.fc(x)
            
            if return_map:
                return x, y
            return x
        else:
            return None, y
            
    def forward(self, x: torch.Tensor, cond=None, return_map=False, corr_layer=None):
        if cond is None:
            raise ValueError("missing parameter 'cond'")
        return self._forward_impl(x, cond, return_map, corr_layer)


def create_conditional_resnet(model_name='resnet152', pretrained=True, **kwargs):
    """Factory function to create a conditional ResNet model"""
    model = CustomResNet(model_name=model_name, pretrained=pretrained, **kwargs)
    return model