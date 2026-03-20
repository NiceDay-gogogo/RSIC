"""
直接从 rsicdtalk.json 词汇表中提取名词并建立同义词映射
"""
import json
import numpy as np
from collections import defaultdict
import argparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


def load_vocab(vocab_file):
    """加载词汇表"""
    with open(vocab_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有词
    words = list(data['ix_to_word'].values())
    # 过滤空词
    words = [w for w in words if w.strip()]
    
    print(f"从词汇表加载了 {len(words)} 个词")
    return words


def is_meaningful_noun(word):
    """判断是否是有意义的名词（非虚词、非形容词、非动词）"""
    # 过滤条件
    if len(word) < 2:  # 太短
        return False
    
    if not word.isalpha():  # 包含非字母
        return False
    
    # 扩展的虚词列表
    meaningless_words = {
        # 冠词、代词、介词
        'the', 'a', 'an', 'this', 'that', 'these', 'those', 'it', 'its',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'in', 'on', 'at', 'to', 'from', 'by', 'with', 'of', 'for',
        'and', 'or', 'but', 'so', 'as', 'if', 'than',
        # 常见形容词
        'big', 'small', 'large', 'little', 'great', 'grand', 'huge', 'tiny',
        'long', 'short', 'wide', 'narrow', 'high', 'low', 'deep', 'shallow',
        'many', 'some', 'several', 'numerous', 'few', 'every', 'all', 'each',
        'red', 'green', 'blue', 'white', 'black', 'gray', 'grey', 'brown',
        'yellow', 'orange', 'purple', 'pink', 'dark', 'light', 'bright',
        'silver', 'golden',
        'good', 'bad', 'new', 'old', 'young', 'beautiful', 'ugly',
        'open', 'closed', 'full', 'empty', 'neat', 'crowded', 'sparse',
        'straight', 'curved', 'round', 'square', 'circular',
        # 常见动词
        'parked', 'arranged', 'surrounded', 'located', 'placed', 'stands',
        'lies', 'sits', 'extends', 'runs', 'flows', 'divide', 'divides',
        'occupies', 'made', 'built', 'constructed',
        # 常见副词
        'very', 'quite', 'rather', 'really', 'just', 'only', 'also',
        'here', 'there', 'where', 'when', 'how', 'why',
        'orderly', 'neatly', 'densely', 'sparsely',
        # 其他虚词
        'next', 'near', 'between', 'around', 'besides', 'front', 'back',
        'side', 'sides', 'middle', 'center', 'either', 'while', 'their',
        'same', 'different', 'others', 'piece', 'pieces', 'area', 'areas',
        'lines', 'line', 'inside', 'outside', 'position', 'space', 'spaces',
        'region', 'regions', 'hand',
        # 数词
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        # 额外需要过滤的词
        'medium', 'bustling', 'spans', 'smart', 'form', 'crossed', 'connects',
        'spray', 'beat', 'against', 'embraced', 'scattered', 'including',
        'size', 'sizes', 'complex', 'differnet',  # 拼写错误
    }
    
    return word.lower() not in meaningless_words


def load_dataset_for_frequency(dataset_file):
    """从数据集中统计词频"""
    print(f"\n加载数据集统计词频: {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    word_freq = defaultdict(int)
    word_image_count = defaultdict(set)
    
    for image in data['images']:
        image_id = image['imgid']
        for sentence in image['sentences']:
            for word in sentence['tokens']:
                word = word.lower()
                word_freq[word] += 1
                word_image_count[word].add(image_id)
    
    # 转换为计数
    word_image_count = {w: len(imgs) for w, imgs in word_image_count.items()}
    
    print(f"统计了 {len(word_freq)} 个词的频率")
    return word_freq, word_image_count


def compute_word_cooccurrence(dataset_file, words_set):
    """计算词的共现矩阵"""
    print(f"\n计算词共现矩阵...")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words_list = sorted(list(words_set))
    word_to_idx = {w: i for i, w in enumerate(words_list)}
    n_words = len(words_list)
    
    cooccurrence = np.zeros((n_words, n_words))
    
    for image in data['images']:
        for sentence in image['sentences']:
            caption_words = [w.lower() for w in sentence['tokens'] if w.lower() in words_set]
            # 计算共现
            for i, w1 in enumerate(caption_words):
                for w2 in caption_words[i:]:
                    idx1 = word_to_idx[w1]
                    idx2 = word_to_idx[w2]
                    cooccurrence[idx1][idx2] += 1
                    if idx1 != idx2:
                        cooccurrence[idx2][idx1] += 1
    
    print(f"计算了 {n_words} 个词的共现矩阵")
    return words_list, cooccurrence


def cluster_words(words_list, cooccurrence, distance_threshold):
    """使用层次聚类发现同义词组"""
    print(f"\n使用层次聚类发现同义词组 (阈值={distance_threshold})...")
    
    # 计算余弦相似度
    similarity = cosine_similarity(cooccurrence)
    
    # 层次聚类
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='average',
        metric='cosine'
    )
    
    labels = clustering.fit_predict(cooccurrence)
    
    # 按聚类分组
    clusters = defaultdict(list)
    for word, label in zip(words_list, labels):
        clusters[label].append(word)
    
    synonym_groups = list(clusters.values())
    print(f"聚类得到 {len(synonym_groups)} 个同义词组")
    
    return synonym_groups


def refine_groups(synonym_groups, word_freq, word_image_count, 
                  min_group_frequency, max_group_size):
    """精炼同义词组"""
    print(f"\n精炼同义词组 (min_freq={min_group_frequency}, max_size={max_group_size})...")
    
    refined_groups = []
    
    for group in synonym_groups:
        # 计算组的总频率
        total_freq = sum(word_freq.get(w, 0) for w in group)
        
        if total_freq < min_group_frequency:
            continue
        
        # 按频率排序
        group_sorted = sorted(group, key=lambda w: word_freq.get(w, 0), reverse=True)
        
        # 限制大小
        if len(group_sorted) > max_group_size:
            group_sorted = group_sorted[:max_group_size]
        
        refined_groups.append({
            'canonical': group_sorted[0],
            'synonyms': group_sorted,
            'total_frequency': total_freq
        })
    
    # 按频率排序
    refined_groups = sorted(refined_groups, key=lambda x: x['total_frequency'], reverse=True)
    
    print(f"精炼后得到 {len(refined_groups)} 个高质量同义词组")
    return refined_groups


def save_results(refined_groups, output_vocab, output_synonyms, output_words, max_attributes):
    """保存结果"""
    # 限制数量
    if len(refined_groups) > max_attributes:
        refined_groups = refined_groups[:max_attributes]
    
    # 生成输出
    vocab = {}
    synonyms = {}
    words = []
    
    for group in refined_groups:
        canonical = group['canonical']
        words.append(canonical)
        vocab[canonical] = {
            'synonyms': group['synonyms'],
            'frequency': group['total_frequency']
        }
        synonyms[canonical] = group['synonyms']
    
    # 保存文件
    print(f"\n保存结果到:")
    print(f"  - {output_vocab}")
    with open(output_vocab, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    
    print(f"  - {output_synonyms}")
    with open(output_synonyms, 'w', encoding='utf-8') as f:
        json.dump(synonyms, f, indent=2, ensure_ascii=False)
    
    print(f"  - {output_words}")
    with open(output_words, 'w', encoding='utf-8') as f:
        json.dump(words, f, indent=2, ensure_ascii=False)
    
    # 打印统计
    print(f"\n{'='*60}")
    print(f"同义词映射统计")
    print(f"{'='*60}")
    print(f"总属性数: {len(words)}")
    print(f"平均每个属性的同义词数: {sum(len(v['synonyms']) for v in vocab.values()) / len(vocab):.1f}")
    
    print(f"\n前20个属性及其同义词:")
    for i, group in enumerate(refined_groups[:20], 1):
        syns = group['synonyms'][:5]
        more = f" (+{len(group['synonyms'])-5} more)" if len(group['synonyms']) > 5 else ""
        print(f"{i:2}. {group['canonical']:15} (freq={group['total_frequency']:5}): {', '.join(syns)}{more}")


def main():
    parser = argparse.ArgumentParser(description='从词汇表直接构建同义词映射')
    parser.add_argument('--vocab_file', type=str, default='data/rsicdtalk.json',
                        help='词汇表文件路径')
    parser.add_argument('--dataset_file', type=str, default='data/dataset_rsicd.json',
                        help='数据集文件路径（用于统计词频和共现）')
    parser.add_argument('--output_vocab', type=str, default='data/attribute_vocab_60.json',
                        help='输出词表文件')
    parser.add_argument('--output_synonyms', type=str, default='data/attribute_synonyms_60.json',
                        help='输出同义词映射文件')
    parser.add_argument('--output_words', type=str, default='data/attribute_words_60.json',
                        help='输出属性词列表文件')
    parser.add_argument('--min_image_count', type=int, default=25,
                        help='词必须出现在至少多少张图片中')
    parser.add_argument('--distance_threshold', type=float, default=0.35,
                        help='层次聚类的距离阈值（越小越严格）')
    parser.add_argument('--min_group_frequency', type=int, default=80,
                        help='同义词组的最小总频率')
    parser.add_argument('--max_group_size', type=int, default=8,
                        help='每个同义词组的最大词数')
    parser.add_argument('--max_attributes', type=int, default=60,
                        help='最多保留多少个属性')
    
    args = parser.parse_args()
    
    # 1. 加载词汇表
    vocab_words = load_vocab(args.vocab_file)
    
    # 2. 过滤出有意义的名词
    print(f"\n过滤出有意义的名词...")
    nouns = [w for w in vocab_words if is_meaningful_noun(w)]
    print(f"过滤后保留 {len(nouns)} 个名词")
    
    # 3. 从数据集统计词频和图片计数
    word_freq, word_image_count = load_dataset_for_frequency(args.dataset_file)
    
    # 4. 过滤低频词
    print(f"\n过滤出现在至少 {args.min_image_count} 张图片中的词...")
    filtered_nouns = [w for w in nouns if word_image_count.get(w.lower(), 0) >= args.min_image_count]
    print(f"过滤后保留 {len(filtered_nouns)} 个高频名词")
    
    # 5. 计算共现矩阵
    words_set = set(w.lower() for w in filtered_nouns)
    words_list, cooccurrence = compute_word_cooccurrence(args.dataset_file, words_set)
    
    # 6. 聚类发现同义词
    synonym_groups = cluster_words(words_list, cooccurrence, args.distance_threshold)
    
    # 7. 精炼同义词组
    refined_groups = refine_groups(
        synonym_groups, word_freq, word_image_count,
        args.min_group_frequency, args.max_group_size
    )
    
    # 8. 保存结果
    save_results(
        refined_groups,
        args.output_vocab,
        args.output_synonyms,
        args.output_words,
        args.max_attributes
    )
    
    print(f"\n完成！")


if __name__ == '__main__':
    main()
