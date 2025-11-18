import torch
import torch.nn as nn
from .backbone import build_backbone
from .head import SiamHead

class SiamTracker(nn.Module):
    def __init__(self, backbone_name="resnet18", feature_dim=256, **kwargs):
        super().__init__()
        self.backbone = build_backbone(backbone_name, feature_dim)
        self.head = SiamHead(feature_dim=feature_dim, **kwargs)
        self.template_feat = None

    @torch.no_grad()
    def set_template(self, template_img):
        """
        template_img: [B, C, H, W] (likely B=1)
        """
        feat = self.backbone(template_img)
        self.template_feat = feat

    def forward(self, search_img, template_img=None):
        """
        During training/validation: pass both template and search.
        During inference: pre-set template via set_template(), pass search only.
        """
        if template_img is not None:
            # Training/validation mode: compute fresh template features
            template_feat = self.backbone(template_img)
        else:
            # Inference mode: use cached template features
            assert self.template_feat is not None, "Call set_template() first or pass template_img"
            template_feat = self.template_feat

        search_feat = self.backbone(search_img)
        cls_logits, bbox_reg = self.head(template_feat, search_feat)
        return cls_logits, bbox_reg
