"""用已训练好的属性模型 + 原始 caption 标签，生成增强版属性标签。

思路：
  - 原始标签 y_true 由 caption 是否包含该词得到，存在漏标（图里有但 caption 没写）。
  - 已训练好的模型在某些图片/属性上预测概率很高（例如 >0.9），很可能是真正的正样本。
  - 我们将标签融合为：
        y_refined = 1， 如果 (y_true == 1) 或 (p_model >= pos_threshold)
        y_refined = 0， 其他情况
  - 只处理 train split（建议），val/test 仍然保留原始标签用于公平评估。

用法示例：
  python refine_attribute_labels_with_model.py \
    --input_json data/rsicd_with_attributes_top20.json \
    --input_att_dir data/rsicdtalk_att \
    --ckpt save/attribute_extractor_top20_focal.pth \
    --split train \
    --pos_threshold 0.9 \
    --output_json data/rsicd_with_attributes_top20_refined.json

然后用 output_json 重新训练：
  python train_attribute_extractor.py \
    --input_json data/rsicd_with_attributes_top20_refined.json \
    ...
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_attribute_extractor import AttributeRSICDDataset, attribute_collate_fn
from models.AttributeFeatureExtractor import AttributeFeatureExtractor


def load_model(ckpt_path: str, num_attributes: int, device: torch.device):
    """加载已训练好的属性模型。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = AttributeFeatureExtractor(
        feat_dim=ckpt_args.get("feat_dim", 2048),
        num_attributes=num_attributes,
        d_model=ckpt_args.get("d_model", 512),
        nhead=ckpt_args.get("nhead", 8),
        num_layers=ckpt_args.get("num_layers", 6),
        dim_feedforward=ckpt_args.get("dim_feedforward", 2048),
        dropout=ckpt_args.get("dropout", 0.1),
        class_weights=None,
        loss_type=ckpt_args.get("loss_type", "bce"),
        focal_gamma=ckpt_args.get("focal_gamma", 2.0),
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, dataloader, device: torch.device):
    """在给定 dataloader 上跑一次模型，返回所有图片的概率预测。"""
    all_probs = []  # List[ [B, A] ]

    for feats, masks, _labels in dataloader:
        feats = feats.to(device)
        masks = masks.to(device)

        logits = model.forward_logits(feats, masks)
        probs = torch.sigmoid(logits).cpu().numpy()  # [B, A]
        all_probs.append(probs)

    if not all_probs:
        return np.zeros((0, 0), dtype=np.float32)

    return np.concatenate(all_probs, axis=0)  # [N, A]


def main():
    parser = argparse.ArgumentParser(description="Refine attribute labels using a trained model")
    parser.add_argument("--input_json", type=str, default="data/rsicd_with_attributes_top20.json",
                        help="JSON with images and attribute_labels (original, caption-based)")
    parser.add_argument("--input_att_dir", type=str, default="data/rsicdtalk_att",
                        help="directory containing pre-extracted att feats (.npz)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="path to trained attribute extractor checkpoint")
    parser.add_argument("--split", type=str, default="train",
                        help="which split to refine: train / val / test")
    parser.add_argument("--pos_threshold", type=float, default=0.9,
                        help="if model prob >= this and GT=0, set refined label to 1")
    parser.add_argument("--output_json", type=str, default="data/rsicd_with_attributes_refined.json",
                        help="path to save refined JSON (will overwrite if exists)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取原始 JSON（包含所有 split）
    print(f"Loading dataset JSON from {args.input_json} ...")
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构建该 split 的 AttributeRSICDDataset
    print(f"Building dataset for split = '{args.split}' ...")
    dataset = AttributeRSICDDataset(args.input_json, args.input_att_dir, split=args.split)
    num_attributes = dataset.num_attributes

    print(f"  Num images in split '{args.split}': {len(dataset)}")
    print(f"  Num attributes: {num_attributes}")

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=attribute_collate_fn,
        pin_memory=True,
    )

    # 加载模型
    print(f"\nLoading model from {args.ckpt} ...")
    model = load_model(args.ckpt, num_attributes, device)

    # 收集概率预测
    print("\nCollecting model predictions on this split ...")
    probs = collect_predictions(model, dataloader, device)  # [N, A]
    if probs.shape[0] != len(dataset):
        raise RuntimeError(f"Number of predictions ({probs.shape[0]}) != dataset size ({len(dataset)})")

    print("  Done. Now refining labels ...")

    # dataset.images 是该 split 的 image dict 列表，和 data['images'] 里的 dict 是同一引用
    # 我们只更新当前 split 的标签，其它 split 保持原样（方便评估）
    pos_thr = float(args.pos_threshold)
    updated_pos = 0

    for idx, img_info in enumerate(dataset.images):
        old_labels = np.array(img_info["attribute_labels"], dtype="float32")  # [A]
        pred_probs = probs[idx]  # [A]
        if old_labels.shape[0] != num_attributes or pred_probs.shape[0] != num_attributes:
            raise RuntimeError(f"Attribute length mismatch at idx {idx}")

        # OR 规则：GT=1 保持 1；GT=0 且 prob>=thr 的设为 1
        refined = old_labels.copy()
        added = (old_labels == 0) & (pred_probs >= pos_thr)
        refined[added] = 1.0

        img_info["attribute_labels"] = refined.astype("float32").tolist()
        updated_pos += int(added.sum())

    print(f"Refinement finished. Added {updated_pos} new positive labels (GT=0 & prob>= {pos_thr}).")

    # 写回 JSON（data 对象已被就地修改，因为 dataset.images 持有的是同一个 dict 引用）
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Refined JSON saved to: {args.output_json}")
    print("\n提示：可以用这个 refined JSON 重新训练模型，作为半监督/噪声修正的一种方式。")


if __name__ == "__main__":
    main()
