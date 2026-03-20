"""分析属性数据集的类别分布和 pos_weight。

用法：
    python analyze_attribute_distribution.py --input_json data/rsicd_with_attributes.json
"""

import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Analyze attribute label distribution")
    parser.add_argument("--input_json", type=str, default="data/rsicd_with_attributes.json",
                        help="JSON with images and attribute_labels")
    parser.add_argument("--vocab_json", type=str, default="data/attribute_vocab.json",
                        help="Optional: JSON with attribute names/vocab")
    parser.add_argument("--split", type=str, default="train",
                        help="Which split to analyze (train/val/test), default=train")
    
    args = parser.parse_args()
    
    # 加载数据
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    images = data["images"]
    split_images = [img for img in images if img.get("split", "train") == args.split]
    
    if len(split_images) == 0:
        print(f"No images found for split '{args.split}'")
        return
    
    print(f"Split: {args.split}")
    print(f"Num images: {len(split_images)}")
    
    # 收集所有标签
    all_labels = []
    for img in split_images:
        labels = img.get("attribute_labels", [])
        all_labels.append(labels)
    
    label_mat = np.array(all_labels, dtype="float32")  # [N, A]
    num_samples, num_attributes = label_mat.shape
    
    print(f"Num attributes: {num_attributes}\n")
    
    # 尝试加载属性名称
    attr_names = None
    try:
        with open(args.vocab_json, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            # attribute_vocab.json 的格式可能是 {"ix_to_word": {...}} 或直接是列表
            if "ix_to_word" in vocab_data:
                ix_to_word = vocab_data["ix_to_word"]
                # 转换为列表（假设索引从1开始或0开始）
                max_ix = max(int(k) for k in ix_to_word.keys())
                attr_names = [""] * (max_ix + 1)
                for ix_str, word in ix_to_word.items():
                    attr_names[int(ix_str)] = word
                # 去掉空字符串开头（如果索引从1开始）
                if attr_names[0] == "":
                    attr_names = attr_names[1:]
            elif isinstance(vocab_data, list):
                attr_names = vocab_data
            else:
                attr_names = None
    except Exception as e:
        print(f"Warning: Could not load attribute names from {args.vocab_json}: {e}")
        attr_names = None
    
    # 统计每个属性
    pos_counts = label_mat.sum(axis=0)  # [A]
    neg_counts = num_samples - pos_counts
    pos_ratio = pos_counts / num_samples
    pos_weight = neg_counts / (pos_counts + 1e-6)
    
    # 打印统计表格
    print(f"{'Index':<6} {'Name':<25} {'Pos':<8} {'Neg':<8} {'Pos%':<8} {'pos_weight':<12}")
    print("=" * 85)
    
    for i in range(num_attributes):
        name = attr_names[i] if (attr_names and i < len(attr_names)) else f"attr_{i}"
        # 有些 vocab 可能把属性名存成 list（多词），统一转成字符串以便对齐打印
        if isinstance(name, list):
            name = " ".join(str(x) for x in name)
        else:
            name = str(name)

        print(
            f"{i:<6} {name:<25} {int(pos_counts[i]):<8} {int(neg_counts[i]):<8} "
            f"{pos_ratio[i]*100:<7.2f}% {pos_weight[i]:<12.2f}"
        )
    
    print("\n" + "=" * 85)
    print("Summary:")
    print(f"  Min pos_ratio:    {pos_ratio.min()*100:.2f}% (index {pos_ratio.argmin()})")
    print(f"  Max pos_ratio:    {pos_ratio.max()*100:.2f}% (index {pos_ratio.argmax()})")
    print(f"  Mean pos_ratio:   {pos_ratio.mean()*100:.2f}%")
    print(f"  Median pos_ratio: {np.median(pos_ratio)*100:.2f}%")
    print()
    print(f"  Min pos_weight:   {pos_weight.min():.2f} (index {pos_weight.argmin()})")
    print(f"  Max pos_weight:   {pos_weight.max():.2f} (index {pos_weight.argmax()})")
    print(f"  Mean pos_weight:  {pos_weight.mean():.2f}")
    print(f"  Median pos_weight:{np.median(pos_weight):.2f}")
    
    # 标记在当前 split 中完全没有正样本的属性
    zero_pos = np.where(pos_counts == 0)[0]
    if len(zero_pos) > 0:
        print(f"\n⚠ Attributes with NO positive samples in split '{args.split}':")
        for idx in zero_pos:
            name = attr_names[idx] if (attr_names and idx < len(attr_names)) else f"attr_{idx}"
            print(f"  - {idx}: {name} (pos=0, neg={int(neg_counts[idx])})")

    # 标记极端不平衡的属性（正样本比例 < 5% 或 > 95%）
    extreme_rare = np.where(pos_ratio < 0.05)[0]
    extreme_common = np.where(pos_ratio > 0.95)[0]
    
    if len(extreme_rare) > 0:
        print(f"\n⚠ Extremely rare attributes (pos% < 5%):")
        for idx in extreme_rare:
            name = attr_names[idx] if (attr_names and idx < len(attr_names)) else f"attr_{idx}"
            print(f"  - {idx}: {name} (pos={int(pos_counts[idx])}, {pos_ratio[idx]*100:.2f}%, weight={pos_weight[idx]:.2f})")
    
    if len(extreme_common) > 0:
        print(f"\n⚠ Extremely common attributes (pos% > 95%):")
        for idx in extreme_common:
            name = attr_names[idx] if (attr_names and idx < len(attr_names)) else f"attr_{idx}"
            print(f"  - {idx}: {name} (pos={int(pos_counts[idx])}, {pos_ratio[idx]*100:.2f}%, weight={pos_weight[idx]:.2f})")
    
    # 标记 pos_weight 特别大的（比如 > 20）
    high_weight = np.where(pos_weight > 20.0)[0]
    if len(high_weight) > 0:
        print(f"\n⚠ Attributes with very high pos_weight (> 20):")
        for idx in high_weight:
            name = attr_names[idx] if (attr_names and idx < len(attr_names)) else f"attr_{idx}"
            print(f"  - {idx}: {name} (pos={int(pos_counts[idx])}, {pos_ratio[idx]*100:.2f}%, weight={pos_weight[idx]:.2f})")


if __name__ == "__main__":
    main()
