import torch
import torch.nn as nn
import torchvision.models as models

def build_backbone(backbone_name="resnet18", feature_dim=256, pretrained=True):
    """
    Build a CNN backbone for feature extraction.

    Args:
        backbone_name: Name of the backbone ('resnet18', 'resnet34', 'mobilenet_v3_small')
        feature_dim: Output feature dimension
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        nn.Module: Backbone model that outputs features of shape (B, feature_dim, H, W)
    """

    if backbone_name == "resnet18":
        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Remove the final FC layer and avgpool
        # ResNet18 has layers: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        # We want to keep up to layer3 or layer4 for feature extraction
        backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool and fc

        # Add a projection layer to get desired feature_dim
        # ResNet18 layer4 output: 512 channels
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(512, feature_dim, kernel_size=1),  # 1x1 conv for channel projection
        )

    elif backbone_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = models.resnet34(weights=weights)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(512, feature_dim, kernel_size=1),
        )

    elif backbone_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone_model = models.mobilenet_v3_small(weights=weights)

        # MobileNetV3 features are in backbone_model.features
        backbone = backbone_model.features

        # MobileNetV3 small output: 576 channels
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(576, feature_dim, kernel_size=1),
        )

    elif backbone_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone_model = models.mobilenet_v3_large(weights=weights)
        backbone = backbone_model.features

        # MobileNetV3 large output: 960 channels
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(960, feature_dim, kernel_size=1),
        )

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    return backbone


class FeatureExtractor(nn.Module):
    """
    Wrapper for backbone that extracts features.
    """

    def __init__(self, backbone_name="resnet18", feature_dim=256, pretrained=True):
        super().__init__()
        self.backbone = build_backbone(backbone_name, feature_dim, pretrained)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            features: Feature map of shape (B, feature_dim, H', W')
        """
        return self.backbone(x)
