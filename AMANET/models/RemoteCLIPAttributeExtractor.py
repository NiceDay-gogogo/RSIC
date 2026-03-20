import os

import torch
import torch.nn as nn
import open_clip
from open_clip.model import ModifiedResNet

from models.AttributeFeatureExtractor import AdvancedAttributeFeatureExtractor


class RemoteCLIPAttributeExtractor(nn.Module):
    """RemoteCLIP visual encoder + AttributeFeatureExtractor head.

    输入: 经过 RemoteCLIP preprocess 的图像 [B, 3, H, W]
    流程:
      - 使用 RemoteCLIP 的 encode_image 得到全局图像特征 [B, D]
      - 将其视作单一“区域”，reshape 为 [B, 1, D]
      - 交给 AttributeFeatureExtractor 做 cross-attention + MLP，输出属性 logits

    注意: 这里使用的是全局 embedding，不是 patch 特征；实现简单，先作为第一版 baseline。
    """

    def __init__(
        self,
        num_attributes,
        clip_model_name="ViT-B-32",
        remoteclip_ckpt_path=None,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        class_weights=None,
        loss_type="bce",
        focal_gamma=2.0,
    ):
        super().__init__()

        self.clip_model_name = clip_model_name

        # 初始化 OpenCLIP 模型（不加载预训练，后面用 RemoteCLIP checkpoint 覆盖）
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name)

        if remoteclip_ckpt_path is not None and os.path.exists(remoteclip_ckpt_path):
            ckpt = torch.load(remoteclip_ckpt_path, map_location="cpu")
            msg = clip_model.load_state_dict(ckpt, strict=False)
            print(f"[RemoteCLIPAttributeExtractor] Loaded RemoteCLIP checkpoint from {remoteclip_ckpt_path}")
            print(msg)
        else:
            if remoteclip_ckpt_path is not None:
                print(
                    f"[RemoteCLIPAttributeExtractor] Warning: RemoteCLIP checkpoint not found at {remoteclip_ckpt_path}. "
                    f"Using base OpenCLIP weights for {clip_model_name}."
                )

        self.clip = clip_model

        # 获取区域特征维度：ResNet backbone 取 layer4 输出通道，其它模型取 encode_image 输出维度
        if isinstance(self.clip.visual, ModifiedResNet):
            self.region_feat_dim = self.clip.visual.layer4[-1].conv3.out_channels
        elif hasattr(self.clip.visual, "output_dim"):
            self.region_feat_dim = self.clip.visual.output_dim
        else:
            with torch.no_grad():
                image_size = getattr(self.clip.visual, "image_size", 224)
                dummy = torch.zeros(1, 3, image_size, image_size)
                feat = self.clip.encode_image(dummy)
                self.region_feat_dim = feat.shape[-1]

        # 属性 head，直接复用现有实现
        self.attr_head = AdvancedAttributeFeatureExtractor(
            feat_dim=self.region_feat_dim,
            num_attributes=num_attributes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            class_weights=class_weights,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
        )

    def _extract_resnet_feats(self, images):
        visual = self.clip.visual
        x = visual.conv1(images)
        x = visual.bn1(x)
        x = visual.act1(x)
        x = visual.conv2(x)
        x = visual.bn2(x)
        x = visual.act2(x)
        x = visual.conv3(x)
        x = visual.bn3(x)
        x = visual.act3(x)
        x = visual.avgpool(x)
        x = visual.layer1(x)
        x = visual.layer2(x)
        x = visual.layer3(x)
        x = visual.layer4(x)
        B, C, H, W = x.shape
        feats = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        masks = torch.ones(B, H * W, device=images.device, dtype=torch.float32)
        return feats, masks

    def _extract_region_feats(self, images):
        """通过 RemoteCLIP 提取图像特征。"""
        if isinstance(self.clip.visual, ModifiedResNet):
            feats, masks = self._extract_resnet_feats(images)
        else:
            x = self.clip.encode_image(images)
            if x.dim() == 2:
                feats = x.unsqueeze(1)
            else:
                B = x.size(0)
                feats = x.view(B, -1, x.shape[-1])
            B, K, _ = feats.size()
            masks = torch.ones(B, K, device=images.device, dtype=torch.float32)
        return feats, masks

    def forward_logits(self, images):
        feats, masks = self._extract_region_feats(images)
        logits, _ = self.attr_head.forward_logits(feats, img_masks=masks)
        return logits

    def forward(self, images):
        logits = self.forward_logits(images)
        probs = torch.sigmoid(logits)
        return probs

    def compute_loss(self, images, targets, reduction="mean"):
        feats, masks = self._extract_region_feats(images)
        loss = self.attr_head.compute_loss(
            feats,
            targets,
            img_masks=masks,
            class_weights=None,
            reduction=reduction,
        )
        return loss

    def set_backbone_trainable(self, trainable: bool):
        for param in self.clip.parameters():
            param.requires_grad = trainable
