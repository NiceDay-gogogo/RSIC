# Script to generate attribute vocabulary from RSICD dataset
# Attributes are defined as nouns that appear more than 50 times in the dataset

import json
import argparse
from collections import Counter

def is_meaningful_noun(word):
    """
    判断词汇是否为有意义的名词，剔除非名词和虚词
    """
    # 常见的非名词词汇（形容词、动词、副词、介词、连词等）
    non_noun_words = {
        'green', 'many', 'some', 'near', 'several', 'around', 'there', 'white', 
        'next', 'parked', 'yellow', 'large', 'dense', 'residential', 'industrial',
        'bareland', 'surrounded', 'curved', 'pieces', 'planted', 'red', 'commercial','irregular','small','beside','dark','fields','building','side','between','gray','blue','basketball','football'
    }
    
    # 常见的无实际意义的虚词
    meaningless_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at','lot', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'theirs',
        'he', 'him', 'his', 'she', 'her', 'hers', 'we', 'us', 'our', 'ours',
        'i', 'me', 'my', 'mine', 'you', 'your', 'yours',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'piece', 'sides', 'area', 'irregular','which'
    }
    
    # 如果词汇在非名词列表中，返回False
    if word in non_noun_words:
        return False
    
    # 如果词汇在无意义词汇列表中，返回False
    if word in meaningless_words:
        return False
    
    # 如果词汇长度小于2，通常无意义
    if len(word) < 2:
        return False
    
    return True

def main(opt):
    # Load the dataset
    print(f"Loading dataset from {opt.input_json}")
    with open(opt.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Count in how many IMAGES each word appears (image-level frequency)
    # 对于每张图片，同一个词只计数一次，避免一张图重复描述同一词导致偏高
    image_counter = Counter()

    # Process all images in the dataset
    for image in data['images']:
        # 收集该图片所有 caption 中出现的词（去重、转小写）
        image_words = set()
        for sentence in image['sentences']:
            words = sentence['raw'].split()
            for w in words:
                image_words.add(w.lower())
        # 每个词在该图片中只计一次
        image_counter.update(image_words)

    print(f"Total unique words (by image-level): {len(image_counter)}")

    # Filter words that appear in more than 'threshold' IMAGES and are meaningful nouns
    threshold = opt.min_count
    attribute_vocab = {}

    for word_lower, img_count in image_counter.items():
        # word_lower 已经是小写
        if img_count > threshold and is_meaningful_noun(word_lower):
            attribute_vocab[word_lower] = img_count
    
    # Sort by frequency (descending)
    sorted_attributes = sorted(attribute_vocab.items(), key=lambda x: x[1], reverse=True)
    
    # 如果指定了最大属性数量，截取前N个
    if opt.max_attributes > 0:
        sorted_attributes = sorted_attributes[:opt.max_attributes]
    
    # Save attribute vocabulary
    print(f"Found {len(sorted_attributes)} attributes that appear more than {threshold} times")
    
    # Save to file
    with open(opt.output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_attributes, f, ensure_ascii=False, indent=2)
    
    print(f"Attribute vocabulary saved to {opt.output_file}")
    
    # Also save just the words (without counts) for easier use
    attribute_words = [word for word, count in sorted_attributes]
    with open(opt.output_words_file, 'w', encoding='utf-8') as f:
        json.dump(attribute_words, f, ensure_ascii=False, indent=2)
    
    print(f"Attribute words list saved to {opt.output_words_file}")
    
    # Print some statistics
    print(f"\nTop {min(40, len(sorted_attributes))} attributes:")
    for i, (word, count) in enumerate(sorted_attributes[:40]):
        print(f"{i+1:2d}. {word:<15} ({count} times)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate attribute vocabulary from RSICD dataset')
    parser.add_argument('--input_json', type=str, default='data/rsicdtalk.json',
                        help='path to the json file containing the dataset')
    parser.add_argument('--output_file', type=str, default='data/attribute_vocab.json',
                        help='path to save the attribute vocabulary with counts')
    parser.add_argument('--output_words_file', type=str, default='data/attribute_words.json',
                        help='path to save the attribute words list')
    parser.add_argument('--min_count', type=int, default=50,
                        help='minimum count for a word to be considered an attribute')
    parser.add_argument('--max_attributes', type=int, default=40,
                        help='maximum number of attributes to include (0 for all)')
    
    opt = parser.parse_args()
    main(opt)