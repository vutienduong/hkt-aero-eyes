import torch
import torch.nn as nn
import torch.nn.functional as F

class SiamHead(nn.Module):
    """
    Siamese detection head for object tracking.

    Computes similarity between template and search features,
    then predicts objectness score and bounding box for each location.
    """

    def __init__(self, feature_dim=256, hidden_dim=256):
        super().__init__()
        self.feature_dim = feature_dim

        # Cross-correlation will produce a similarity map
        # Then we use conv layers to predict classification and bbox regression

        # Classification head (objectness)
        self.cls_head = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),  # 1 channel: object vs background
        )

        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 4, kernel_size=1),  # 4 values: (x1, y1, x2, y2) or (dx, dy, dw, dh)
        )

    def cross_correlation(self, template_feat, search_feat):
        """
        Compute cross-correlation between template and search features.

        Args:
            template_feat: (B, C, H_t, W_t) - template features
            search_feat: (B, C, H_s, W_s) - search features

        Returns:
            correlation_map: (B, C, H_s, W_s) - correlation map
        """
        # Simple approach: element-wise multiply after broadcasting
        # More sophisticated: use depthwise cross-correlation

        # For simplicity, we'll use adaptive avg pooling to get a single template vector
        # then broadcast and multiply
        B, C, H_s, W_s = search_feat.shape

        # Pool template to single vector per channel
        template_vec = F.adaptive_avg_pool2d(template_feat, (1, 1))  # (B, C, 1, 1)

        # Broadcast and multiply
        correlation = search_feat * template_vec  # (B, C, H_s, W_s)

        return correlation

    def forward(self, template_feat, search_feat):
        """
        Args:
            template_feat: Template features (B, C, H_t, W_t)
            search_feat: Search features (B, C, H_s, W_s)

        Returns:
            cls_logits: Classification logits (B, 1, H_s, W_s)
            bbox_pred: Bounding box predictions (B, 4, H_s, W_s)
        """
        # Compute correlation
        corr_feat = self.cross_correlation(template_feat, search_feat)

        # Predict classification and bbox
        cls_logits = self.cls_head(corr_feat)  # (B, 1, H_s, W_s)
        bbox_pred = self.bbox_head(corr_feat)  # (B, 4, H_s, W_s)

        return cls_logits, bbox_pred


class SiamRPNHead(nn.Module):
    """
    More sophisticated Siamese RPN-style head with anchor-based detection.
    """

    def __init__(self, feature_dim=256, hidden_dim=256, num_anchors=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_anchors = num_anchors

        # Anchor-based prediction
        # Classification: num_anchors * 2 (object/background for each anchor)
        self.cls_head = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_anchors * 2, kernel_size=1),
        )

        # Bbox regression: num_anchors * 4 (dx, dy, dw, dh for each anchor)
        self.bbox_head = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_anchors * 4, kernel_size=1),
        )

    def cross_correlation(self, template_feat, search_feat):
        """Similar to SiamHead"""
        template_vec = F.adaptive_avg_pool2d(template_feat, (1, 1))
        correlation = search_feat * template_vec
        return correlation

    def forward(self, template_feat, search_feat):
        """
        Returns:
            cls_logits: (B, num_anchors*2, H, W)
            bbox_pred: (B, num_anchors*4, H, W)
        """
        corr_feat = self.cross_correlation(template_feat, search_feat)

        cls_logits = self.cls_head(corr_feat)
        bbox_pred = self.bbox_head(corr_feat)

        return cls_logits, bbox_pred
