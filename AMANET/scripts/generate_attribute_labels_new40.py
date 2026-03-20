"""
根据40个属性词和同义词映射表，为数据集生成属性标签

使用方法:
    python scripts/generate_attribute_labels_40.py \
        --input_json data/dataset_rsicd.json \
        --attribute_words data/attribute_words_40.json \
        --attribute_synonyms data/attribute_synonyms_40.json \
        --output_json data/rsicd_with_attributes_40.json
"""

import json
import argparse
from collections import defaultdict


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_word_to_attribute_map(attribute_words, attribute_synonyms):
    """
    构建从词汇到属性索引的映射
    
    Args:
        attribute_words: 40个属性词列表
        attribute_synonyms: 同义词映射字典 {属性词: [同义词列表]}
    
    Returns:
        word_to_attr_idx: {词汇: 属性索引}
    """
    word_to_attr_idx = {}
    
    for idx, attr_word in enumerate(attribute_words):
        # 获取该属性的所有同义词
        synonyms = attribute_synonyms.get(attr_word, [attr_word])
        
        for syn in synonyms:
            syn_lower = syn.lower()
            if syn_lower not in word_to_attr_idx:
                word_to_attr_idx[syn_lower] = idx
            # 如果已存在，说明多个属性共享同一个同义词，保留第一个
    
    return word_to_attr_idx


def generate_labels_for_image(image_info, word_to_attr_idx, num_attributes):
    """
    为单张图像生成属性标签
    
    Args:
        image_info: 图像信息字典，包含 'sentences' 字段
        word_to_attr_idx: 词汇到属性索引的映射
        num_attributes: 属性总数
    
    Returns:
        labels: [num_attributes] 的0/1列表
    """
    labels = [0] * num_attributes
    
    # 收集该图像所有描述中的词汇
    all_words = set()
    for sentence in image_info.get('sentences', []):
        tokens = sentence.get('tokens', [])
        for token in tokens:
            all_words.add(token.lower())
    
    # 检查每个词是否对应某个属性
    for word in all_words:
        if word in word_to_attr_idx:
            attr_idx = word_to_attr_idx[word]
            labels[attr_idx] = 1
    
    return labels


def main():
    parser = argparse.ArgumentParser(description='Generate attribute labels for RSICD dataset')
    parser.add_argument('--input_json', type=str, default='data/dataset_rsicd.json',
                        help='输入数据集JSON文件路径')
    parser.add_argument('--attribute_words', type=str, default='data/attribute_words_new40.json',
                        help='40个属性词列表文件路径')
    parser.add_argument('--attribute_synonyms', type=str, default='data/attribute_synonyms_new40.json',
                        help='同义词映射表文件路径')
    parser.add_argument('--output_json', type=str, default='data/rsicd_with_attributes_new40.json',
                        help='输出带属性标签的JSON文件路径')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载数据集: {args.input_json}")
    dataset = load_json(args.input_json)
    
    print(f"加载属性词: {args.attribute_words}")
    attribute_words = load_json(args.attribute_words)
    
    print(f"加载同义词映射: {args.attribute_synonyms}")
    attribute_synonyms = load_json(args.attribute_synonyms)
    
    num_attributes = len(attribute_words)
    print(f"属性数量: {num_attributes}")
    
    # 构建词汇到属性的映射
    word_to_attr_idx = build_word_to_attribute_map(attribute_words, attribute_synonyms)
    print(f"同义词映射总数: {len(word_to_attr_idx)}")
    
    # 为每张图像生成标签
    print("为图像生成属性标签...")
    total_positive = defaultdict(int)  # 统计每个属性的正样本数
    
    for image in dataset['images']:
        labels = generate_labels_for_image(image, word_to_attr_idx, num_attributes)
        image['attribute_labels'] = labels
        
        # 统计
        for idx, label in enumerate(labels):
            if label == 1:
                total_positive[idx] += 1
    
    # 保存结果
    print(f"保存结果到: {args.output_json}")
    save_json(dataset, args.output_json)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("属性标签统计")
    print("=" * 60)
    print(f"{'属性词':<20} {'正样本数':>10} {'比例':>10}")
    print("-" * 60)
    
    total_images = len(dataset['images'])
    for idx, attr_word in enumerate(attribute_words):
        count = total_positive[idx]
        ratio = count / total_images * 100
        print(f"{attr_word:<20} {count:>10} {ratio:>9.2f}%")
    
    print("-" * 60)
    print(f"图像总数: {total_images}")
    print(f"至少有1个属性的图像: {sum(1 for img in dataset['images'] if sum(img['attribute_labels']) > 0)}")
    
    # 保存属性词索引映射（方便后续使用）
    attr_idx_map = {attr: idx for idx, attr in enumerate(attribute_words)}
    idx_attr_map = {idx: attr for idx, attr in enumerate(attribute_words)}
    
    mapping_file = args.output_json.replace('.json', '_mapping.json')
    save_json({
        'attribute_words': attribute_words,
        'attr_to_idx': attr_idx_map,
        'idx_to_attr': idx_attr_map,
        'num_attributes': num_attributes
    }, mapping_file)
    print(f"\n属性映射保存到: {mapping_file}")


if __name__ == '__main__':
    main()
