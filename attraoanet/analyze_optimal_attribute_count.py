"""分析最优属性个数：评估保留不同数量的属性时模型的表现。

使用已训练好的模型，在不同属性子集（按正样本比例从高到低选择）上评估性能。

用法：
    python analyze_optimal_attribute_count.py \
      --ckpt save/attribute_extractor_focal.pth \
      --threshold 0.6
"""

import argparse
import json
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
        
        logits = model.forward_logits(feats, masks)
        probs = torch.sigmoid(logits)
        
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    labels_np = np.concatenate(all_labels, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)
    
    return labels_np, probs_np


def compute_metrics_for_subset(labels, probs, attr_indices, threshold=0.5):
    """计算指定属性子集上的指标。
    
    Args:
        labels: [N, num_attributes] 完整标签
        probs: [N, num_attributes] 完整预测概率
        attr_indices: 要评估的属性索引列表
        threshold: 二值化阈值
    
    Returns:
        f1_micro, f1_macro, map_macro
    """
    # 只取指定属性列
    subset_labels = labels[:, attr_indices]
    subset_probs = probs[:, attr_indices]
    subset_preds = (subset_probs >= threshold).astype(int)
    
    f1_micro = f1_score(subset_labels, subset_preds, average='micro', zero_division=0)
    f1_macro = f1_score(subset_labels, subset_preds, average='macro', zero_division=0)
    map_macro = average_precision_score(subset_labels, subset_probs, average='macro')
    
    return f1_micro, f1_macro, map_macro


def get_attribute_pos_ratios(json_path, split='train'):
    """统计训练集每个属性的正样本比例。"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = [img for img in data['images'] if img.get('split', 'train') == split]
    
    all_labels = []
    for img in images:
        labels = np.array(img['attribute_labels'], dtype='float32')
        all_labels.append(labels)
    
    label_mat = np.stack(all_labels, axis=0)  # [N, A]
    pos_counts = label_mat.sum(axis=0)
    pos_ratios = pos_counts / len(images)
    
    return pos_ratios, pos_counts


def main():
    parser = argparse.ArgumentParser(description="Analyze optimal attribute count")
    parser.add_argument("--input_json", type=str, default="data/rsicd_with_attributes.json")
    parser.add_argument("--input_att_dir", type=str, default="data/rsicdtalk_att")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default="save/attribute_extractor_focal.pth")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Global threshold for binarization")
    parser.add_argument("--min_pos_ratio", type=float, default=0.05,
                        help="Minimum positive ratio for recommended subset")
    
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
    
    num_attributes = val_dataset.num_attributes
    print(f"Total attributes: {num_attributes}")
    print(f"Val images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}\n")
    
    # 加载模型
    ckpt = torch.load(args.ckpt, map_location=device)
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
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"], strict=False)
    
    # 获取训练集的属性正样本比例（用于排序）
    print("Analyzing attribute distribution on training set...")
    pos_ratios, pos_counts = get_attribute_pos_ratios(args.input_json, split='train')
    
    # 按正样本比例从高到低排序属性
    sorted_indices = np.argsort(pos_ratios)[::-1]  # 降序
    
    print("\nAttribute ranking by positive ratio (train set):")
    print(f"{'Rank':<6} {'Index':<8} {'Pos_Count':<12} {'Pos_Ratio':<12}")
    print("=" * 45)
    for rank, idx in enumerate(sorted_indices[:20], 1):  # 只打印前20个
        print(f"{rank:<6} {idx:<8} {int(pos_counts[idx]):<12} {pos_ratios[idx]*100:<11.2f}%")
    print("...\n")
    
    # 收集测试集预测
    print("Collecting predictions on test set...")
    test_labels, test_probs = collect_predictions(model, test_loader, device)
    
    # 评估不同 K 值（top-K 最常见属性）
    K_values = [5, 10, 15, 20, 25, 30, 35, 40]
    K_values = [k for k in K_values if k <= num_attributes]
    
    print(f"\nEvaluating performance with different attribute counts (threshold={args.threshold}):")
    print(f"{'K':<6} {'F1_micro':<12} {'F1_macro':<12} {'mAP_macro':<12}")
    print("=" * 48)
    
    results = []
    for K in K_values:
        top_K_indices = sorted_indices[:K]
        f1_micro, f1_macro, map_macro = compute_metrics_for_subset(
            test_labels, test_probs, top_K_indices, threshold=args.threshold
        )
        results.append({
            'K': K,
            'F1_micro': f1_micro,
            'F1_macro': f1_macro,
            'mAP_macro': map_macro
        })
        print(f"{K:<6} {f1_micro:<12.4f} {f1_macro:<12.4f} {map_macro:<12.4f}")
    
    # 找到 F1_macro 最高的 K
    best_result = max(results, key=lambda x: x['F1_macro'])
    print(f"\n{'='*48}")
    print(f"Best K by F1_macro: {best_result['K']}")
    print(f"  F1_micro  = {best_result['F1_micro']:.4f}")
    print(f"  F1_macro  = {best_result['F1_macro']:.4f}")
    print(f"  mAP_macro = {best_result['mAP_macro']:.4f}")
    
    # 生成推荐属性子集（基于 min_pos_ratio）
    recommended_indices = [idx for idx in sorted_indices if pos_ratios[idx] >= args.min_pos_ratio]
    
    print(f"\n{'='*48}")
    print(f"Recommended attribute subset (pos_ratio >= {args.min_pos_ratio*100:.1f}%):")
    print(f"  Number of attributes: {len(recommended_indices)}")
    print(f"  Attribute indices: {recommended_indices}")
    
    # 评估推荐子集的性能
    if len(recommended_indices) > 0:
        f1_micro, f1_macro, map_macro = compute_metrics_for_subset(
            test_labels, test_probs, recommended_indices, threshold=args.threshold
        )
        print(f"\n  Performance on test set:")
        print(f"    F1_micro  = {f1_micro:.4f}")
        print(f"    F1_macro  = {f1_macro:.4f}")
        print(f"    mAP_macro = {map_macro:.4f}")
    
    # 对比全部 40 个属性
    if len(recommended_indices) < num_attributes:
        all_indices = list(range(num_attributes))
        f1_micro_all, f1_macro_all, map_macro_all = compute_metrics_for_subset(
            test_labels, test_probs, all_indices, threshold=args.threshold
        )
        print(f"\n  Performance on test set (all {num_attributes} attributes for comparison):")
        print(f"    F1_micro  = {f1_micro_all:.4f}")
        print(f"    F1_macro  = {f1_macro_all:.4f}")
        print(f"    mAP_macro = {map_macro_all:.4f}")
        
        if len(recommended_indices) > 0:
            print(f"\n  Improvement by using recommended subset:")
            print(f"    F1_micro: {f1_micro_all:.4f} -> {f1_micro:.4f} ({(f1_micro - f1_micro_all)*100:+.2f}%)")
            print(f"    F1_macro: {f1_macro_all:.4f} -> {f1_macro:.4f} ({(f1_macro - f1_macro_all)*100:+.2f}%)")
            print(f"    mAP_macro: {map_macro_all:.4f} -> {map_macro:.4f} ({(map_macro - map_macro_all)*100:+.2f}%)")
    
    # 保存推荐的属性索引到文件
    output_file = "recommended_attribute_indices.json"
    with open(output_file, 'w') as f:
        json.dump({
            'recommended_indices': [int(x) for x in recommended_indices],
            'min_pos_ratio': float(args.min_pos_ratio),
            'num_attributes': int(len(recommended_indices)),
            'all_sorted_indices': [int(x) for x in sorted_indices.tolist()],
            'pos_ratios': {int(idx): float(pos_ratios[idx]) for idx in range(num_attributes)}
        }, f, indent=2)
    print(f"\nRecommended attribute indices saved to: {output_file}")


if __name__ == "__main__":
    main()
