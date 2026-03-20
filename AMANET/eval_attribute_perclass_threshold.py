"""为每个属性单独找最优阈值（基于验证集），然后在测试集上评估。

用法：
    python eval_attribute_perclass_threshold.py --ckpt save/attribute_extractor.pth
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score

from train_attribute_extractor import AttributeRSICDDataset, attribute_collate_fn
from models.AttributeFeatureExtractor import AttributeFeatureExtractor


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """收集模型在整个数据集上的预测概率和真实标签。"""
    model.eval()
    
    all_labels = []
    all_probs = []
    
    for feats, masks, labels in dataloader:
        feats = feats.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        logits = model.forward_logits(feats, masks)
        probs = torch.sigmoid(logits)
        
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    labels_np = np.concatenate(all_labels, axis=0)  # [N, num_attributes]
    probs_np = np.concatenate(all_probs, axis=0)
    
    return labels_np, probs_np


def find_optimal_thresholds_per_class(labels, probs, thresholds=None):
    """为每个属性找最优阈值（基于 F1 score）。
    
    Args:
        labels: [N, A] 真实标签
        probs: [N, A] 预测概率
        thresholds: 候选阈值列表
    
    Returns:
        best_thresholds: [A] 每个属性的最优阈值
        per_class_f1: [A] 每个属性的最优 F1
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    num_attributes = labels.shape[1]
    best_thresholds = np.zeros(num_attributes)
    per_class_f1 = np.zeros(num_attributes)
    
    for attr_idx in range(num_attributes):
        attr_labels = labels[:, attr_idx]
        attr_probs = probs[:, attr_idx]
        
        best_f1 = -1.0
        best_thr = 0.5
        
        for thr in thresholds:
            attr_preds = (attr_probs >= thr).astype(int)
            f1 = f1_score(attr_labels, attr_preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        
        best_thresholds[attr_idx] = best_thr
        per_class_f1[attr_idx] = best_f1
    
    return best_thresholds, per_class_f1


def evaluate_with_perclass_thresholds(labels, probs, thresholds):
    """用 per-class 阈值评估。"""
    num_samples, num_attributes = labels.shape
    preds = np.zeros_like(labels, dtype=int)
    
    for attr_idx in range(num_attributes):
        thr = thresholds[attr_idx]
        preds[:, attr_idx] = (probs[:, attr_idx] >= thr).astype(int)
    
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    map_macro = average_precision_score(labels, probs, average='macro')
    
    return f1_micro, f1_macro, map_macro


def main():
    parser = argparse.ArgumentParser(description="Per-class threshold optimization for attribute extractor")
    parser.add_argument("--input_json", type=str, default="data/rsicd_with_attributes.json")
    parser.add_argument("--input_att_dir", type=str, default="data/rsicdtalk_att")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default="save/attribute_extractor.pth")
    parser.add_argument("--threshold_step", type=float, default=0.05,
                        help="Step size for threshold search (default: 0.05)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载验证集和测试集
    val_dataset = AttributeRSICDDataset(args.input_json, args.input_att_dir, split="val")
    test_dataset = AttributeRSICDDataset(args.input_json, args.input_att_dir, split="test")
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=attribute_collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=attribute_collate_fn, pin_memory=True
    )
    
    print(f"Val images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Num attributes: {val_dataset.num_attributes}\n")
    
    # 加载模型
    ckpt = torch.load(args.ckpt, map_location=device)
    num_attributes = ckpt.get("num_attributes", val_dataset.num_attributes)
    ckpt_args = ckpt.get("args", {})
    
    feat_dim = ckpt_args.get("feat_dim", 2048)
    d_model = ckpt_args.get("d_model", 512)
    nhead = ckpt_args.get("nhead", 8)
    num_layers = ckpt_args.get("num_layers", 6)
    dim_feedforward = ckpt_args.get("dim_feedforward", 2048)
    dropout = ckpt_args.get("dropout", 0.1)
    
    model = AttributeFeatureExtractor(
        feat_dim=feat_dim,
        num_attributes=num_attributes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        class_weights=None,
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"], strict=False)
    
    # 在验证集上收集预测
    print("Collecting predictions on validation set...")
    val_labels, val_probs = collect_predictions(model, val_loader, device)
    
    # 为每个属性找最优阈值
    print(f"Searching optimal thresholds (step={args.threshold_step})...")
    thresholds_search = np.arange(0.1, 1.0, args.threshold_step)
    best_thresholds, per_class_f1 = find_optimal_thresholds_per_class(
        val_labels, val_probs, thresholds=thresholds_search
    )
    
    # 打印每个属性的最优阈值
    print("\nPer-class optimal thresholds (based on validation set):")
    print(f"{'Index':<6} {'Best_Thr':<10} {'Val_F1':<10}")
    print("=" * 30)
    for i in range(num_attributes):
        print(f"{i:<6} {best_thresholds[i]:<10.2f} {per_class_f1[i]:<10.4f}")
    
    print(f"\nMean optimal threshold: {best_thresholds.mean():.3f}")
    print(f"Std optimal threshold:  {best_thresholds.std():.3f}")
    print(f"Min optimal threshold:  {best_thresholds.min():.3f}")
    print(f"Max optimal threshold:  {best_thresholds.max():.3f}")
    
    # 在验证集上评估（sanity check）
    val_f1_micro, val_f1_macro, val_map = evaluate_with_perclass_thresholds(
        val_labels, val_probs, best_thresholds
    )
    print(f"\nValidation set (with per-class thresholds):")
    print(f"  F1_micro  = {val_f1_micro:.4f}")
    print(f"  F1_macro  = {val_f1_macro:.4f}")
    print(f"  mAP_macro = {val_map:.4f}")
    
    # 在测试集上收集预测并评估
    print("\nCollecting predictions on test set...")
    test_labels, test_probs = collect_predictions(model, test_loader, device)
    
    test_f1_micro, test_f1_macro, test_map = evaluate_with_perclass_thresholds(
        test_labels, test_probs, best_thresholds
    )
    
    print(f"\nTest set (with per-class thresholds):")
    print(f"  F1_micro  = {test_f1_micro:.4f}")
    print(f"  F1_macro  = {test_f1_macro:.4f}")
    print(f"  mAP_macro = {test_map:.4f}")
    
    # 对比：用单一阈值 0.5 的效果
    test_preds_05 = (test_probs >= 0.5).astype(int)
    test_f1_micro_05 = f1_score(test_labels, test_preds_05, average='micro', zero_division=0)
    test_f1_macro_05 = f1_score(test_labels, test_preds_05, average='macro', zero_division=0)
    
    print(f"\nTest set (with global threshold=0.5, for comparison):")
    print(f"  F1_micro  = {test_f1_micro_05:.4f}")
    print(f"  F1_macro  = {test_f1_macro_05:.4f}")
    
    print(f"\nImprovement by using per-class thresholds:")
    print(f"  F1_micro: {test_f1_micro_05:.4f} -> {test_f1_micro:.4f} ({(test_f1_micro - test_f1_micro_05)*100:+.2f}%)")
    print(f"  F1_macro: {test_f1_macro_05:.4f} -> {test_f1_macro:.4f} ({(test_f1_macro - test_f1_macro_05)*100:+.2f}%)")


if __name__ == "__main__":
    main()
