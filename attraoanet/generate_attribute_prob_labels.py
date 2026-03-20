#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用属性特征提取器为 RSICD 数据生成概率版属性标注。

流程：
1. 读取原始 rsicd_with_attributes_xxx.json 和区域特征（att_feats）。
2. 使用训练好的 AttributeFeatureExtractor 推理得到 40 维属性概率。
3. 对正例属性做随机区间提升，弱化低分正例（与可视化脚本一致的处理逻辑）。
4. 将处理后的概率写入新的 json 文件中：
     - attribute_labels: 覆盖为概率值（兼容现有 DataLoader）。
     - attribute_probs:  同样保存概率值，方便区分。
     - attribute_binary_labels: 备份原始 0/1 标签。
"""

import argparse
import json
import os
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_attribute_extractor import AttributeRSICDDataset, attribute_collate_fn
from models.AttributeFeatureExtractor import AttributeFeatureExtractor


def build_model(ckpt_path: str, device: torch.device) -> AttributeFeatureExtractor:
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = AttributeFeatureExtractor(
        feat_dim=ckpt_args.get("feat_dim", 2048),
        num_attributes=ckpt_args.get("num_attributes", 40),
        d_model=ckpt_args.get("d_model", 512),
        nhead=ckpt_args.get("nhead", 8),
        num_layers=ckpt_args.get("num_layers", 6),
        dim_feedforward=ckpt_args.get("dim_feedforward", 2048),
        dropout=ckpt_args.get("dropout", 0.1),
        class_weights=None,
        loss_type=ckpt_args.get("loss_type", "focal"),
        focal_gamma=ckpt_args.get("focal_gamma", 2.0),
        use_feature_enhancement=ckpt_args.get("use_feature_enhancement", True),
        use_relative_pos=ckpt_args.get("use_relative_pos", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model


def adjust_probabilities(probs: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    对正例概率做随机提升，对异常高的负例做抑制，确保正例明显高于负例。
    probs/labels shape: [B, A]
    """
    adjusted = probs.copy()
    pos_mask = labels >= 0.5
    neg_mask = ~pos_mask

    high_pos = pos_mask & (adjusted >= 0.8)
    if high_pos.any():
        adjusted_high = rng.uniform(0.9, 0.97, size=high_pos.sum())
        adjusted[high_pos] = np.maximum(adjusted[high_pos], adjusted_high)

    mid_pos = pos_mask & (adjusted >= 0.7) & (adjusted < 0.8)
    if mid_pos.any():
        adjusted[mid_pos] = rng.uniform(0.8, 0.87, size=mid_pos.sum())

    low_pos = pos_mask & (adjusted < 0.7)
    if low_pos.any():
        adjusted[low_pos] = rng.uniform(0.7, 0.78, size=low_pos.sum())

    high_neg = neg_mask & (adjusted >= 0.7)
    if high_neg.any():
        adjusted[high_neg] = rng.uniform(0.35, 0.6, size=high_neg.sum())

    return adjusted


def get_image_id(entry: Dict[str, Any]) -> str:
    if "imgid" in entry:
        return str(entry["imgid"])
    if "id" in entry:
        return str(entry["id"])
    raise KeyError("Image entry must contain 'imgid' or 'id'")


def process_split(
    split: str,
    dataset: AttributeRSICDDataset,
    model: AttributeFeatureExtractor,
    device: torch.device,
    master_lookup: Dict[str, Dict[str, Any]],
    rng: np.random.Generator,
    apply_adjustment: bool,
) -> int:
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=attribute_collate_fn,
        num_workers=0,
    )
    processed = 0
    with torch.no_grad():
        offset = 0
        for feats, masks, labels in loader:
            feats = feats.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            logits, _ = model.forward_logits(feats, masks)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()

            if apply_adjustment:
                probs = adjust_probabilities(probs, labels_np, rng)

            batch_size = probs.shape[0]
            for i in range(batch_size):
                img_info = dataset.images[offset + i]
                img_id = get_image_id(img_info)
                master_entry = master_lookup.get(img_id)
                if master_entry is None:
                    continue

                original = master_entry.get("attribute_labels", None)
                if original is not None and "attribute_binary_labels" not in master_entry:
                    master_entry["attribute_binary_labels"] = original

                prob_list = probs[i].tolist()
                master_entry["attribute_labels"] = prob_list
                master_entry["attribute_probs"] = prob_list
                processed += 1

            offset += batch_size
    print(f"[{split}] processed {processed} images.")
    return processed


def main():
    parser = argparse.ArgumentParser(description="Generate probabilistic attribute labels.")
    parser.add_argument("--input_json", type=str, default="data/rsicd_with_attributes_new40.json")
    parser.add_argument("--input_att_dir", type=str, default="data/rsicdtalk_att")
    parser.add_argument("--ckpt", type=str, required=True, help="Attribute extractor checkpoint (.pth)")
    parser.add_argument("--output_json", type=str, default="data/rsicd_with_attr_probs_new40.json")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="Comma-separated splits to process")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply_adjustment", action="store_true",
                        help="启用真值概率随机提升逻辑")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    info = json.load(open(args.input_json, "r"))
    master_lookup = {}
    for img in info["images"]:
        img_id = get_image_id(img)
        master_lookup[img_id] = img

    model = build_model(args.ckpt, device)
    rng = np.random.default_rng(args.seed)

    total = 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        try:
            dataset = AttributeRSICDDataset(args.input_json, args.input_att_dir, split=split)
        except ValueError:
            print(f"[{split}] split not found, skipping.")
            continue
        total += process_split(split, dataset, model, device, master_lookup, rng, args.apply_adjustment)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(info, f)
    print(f"Saved probabilistic annotations for {total} images to {args.output_json}")


if __name__ == "__main__":
    main()
