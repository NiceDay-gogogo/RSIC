#!/usr/bin/env python3
"""
Generate attribute vocab, per-image multi-labels, and adjacency matrix for GCN from caption data.
- Attribute vocab: top-K frequent tokens (excluding specials).
- Labels: y_attr.npy shape [num_images, K], 1 if attribute appears in any caption of that image.
- Adjacency: PMI-positive co-occurrence matrix saved as adj.npy.
Outputs are placed under data/<dataset>/attr/.
"""
import argparse
import json
import os
import math
import numpy as np
from collections import Counter, defaultdict


SPECIAL_TOKENS = {"<pad>", "<start>", "<end>", "<unk>"}
# Extended stopwords: basic function words + verbs/adverbs/prepositions that are not visual attributes
STOPWORDS = {
    # Articles, conjunctions, prepositions
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "there",
    "it", "as", "for", "from", "into", "over", "under", "above", "below",
    "no", "not", "none", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "some", "any", "each", "other", "others", "both",
    "all", "many", "few", "more", "most", "much", "very", "just", "has", "have",
    "had", "can", "could", "should", "would", "will", "may", "might", "lot", "lots",
    # Pronouns
    "them", "they", "their", "its", "those", "these", "which", "what", "who",
    "another", "something", "nothing", "everything",
    # Common verbs (non-visual)
    "go", "goes", "going", "went", "gone", "come", "comes", "coming", "came",
    "make", "makes", "made", "making", "do", "does", "did", "done", "doing",
    "get", "gets", "got", "getting", "take", "takes", "took", "taken", "taking",
    "put", "puts", "putting", "give", "gives", "gave", "given", "giving",
    "see", "sees", "saw", "seen", "seeing", "look", "looks", "looked", "looking",
    "seem", "seems", "seemed", "seeming", "appear", "appears", "appeared",
    "throw", "throws", "threw", "thrown", "throwing", "beat", "beats", "beating",
    "melt", "melts", "melted", "melting", "slapping", "pressed", "compose", "constitute",
    # Adverbs
    "neatly", "densely", "haphazardly", "vertically", "diagonally", "scatteredly",
    "violently", "compactly", "dispersedly", "triangularly", "peacefully", "quietly",
    "only", "again", "here", "also", "still", "already", "always", "never", "ever",
    "forward", "up", "down", "out", "away",
    # Prepositions/conjunctions
    "beside", "through", "across", "between", "around", "while", "without", "next",
    # Size/quantity modifiers (keep some like 'big', 'small' but remove vague ones)
    "same", "different", "number",
    # Common typos in your dataset
    "surounded", "arround", "sorrounded", "formland", "foasm", "foams", "plamts",
    "roofss", "ars", "spase", "wothe", "coursea", "flurish", "fo", "containters",
}


def load_wordmap(path):
    with open(path, "r") as f:
        return json.load(f)


def top_k_attrs(captions, idx2word, k):
    cnt = Counter()
    for cap in captions:
        for wid in cap:
            w = idx2word.get(str(wid), idx2word.get(wid, None))
            if w is None:
                continue
            w_l = w.lower()
            if w_l in SPECIAL_TOKENS or w_l in STOPWORDS or len(w_l.strip()) <= 1:
                continue
            cnt[w_l] += 1
    most_common = [w for w, _ in cnt.most_common(k)]
    return most_common


def build_labels(captions, cpi, idx2word, attr_vocab):
    vocab_index = {w: i for i, w in enumerate(attr_vocab)}
    num_images = len(captions) // cpi
    labels = np.zeros((num_images, len(attr_vocab)), dtype=np.float32)
    for img_idx in range(num_images):
        start = img_idx * cpi
        end = start + cpi
        seen = set()
        for cap in captions[start:end]:
            for wid in cap:
                w = idx2word.get(str(wid), idx2word.get(wid, None))
                if w is None:
                    continue
                w_l = w.lower()
                if w_l in vocab_index:
                    seen.add(vocab_index[w_l])
        if seen:
            labels[img_idx, list(seen)] = 1.0
    return labels


def build_pmi_adj(captions, idx2word, attr_vocab, eps=1e-8):
    vocab_index = {w: i for i, w in enumerate(attr_vocab)}
    co = np.zeros((len(attr_vocab), len(attr_vocab)), dtype=np.float64)
    occur = np.zeros((len(attr_vocab),), dtype=np.float64)
    for cap in captions:
        tokens = set()
        for wid in cap:
            w = idx2word.get(str(wid), idx2word.get(wid, None))
            if w is None:
                continue
            w_l = w.lower()
            if w_l in vocab_index:
                tokens.add(vocab_index[w_l])
        tokens = list(tokens)
        for i in range(len(tokens)):
            occur[tokens[i]] += 1
            for j in range(i + 1, len(tokens)):
                co[tokens[i], tokens[j]] += 1
                co[tokens[j], tokens[i]] += 1
    total = len(captions)
    adj = np.zeros_like(co)
    for i in range(len(attr_vocab)):
        for j in range(len(attr_vocab)):
            if i == j:
                continue
            p_ij = co[i, j] / total
            if p_ij <= 0:
                continue
            p_i = occur[i] / total
            p_j = occur[j] / total
            pmi = math.log(p_ij / (p_i * p_j + eps) + eps)
            if pmi > 0:
                adj[i, j] = pmi
    # row-normalize
    row_sum = adj.sum(axis=1, keepdims=True) + eps
    adj = adj / row_sum
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", required=True, help="e.g., ./data/UCM")
    parser.add_argument("--data_name", required=True, help="e.g., UCM_5_cap_per_img_4_min_word_freq")
    parser.add_argument("--top_k", type=int, default=500, help="number of attributes to keep")
    parser.add_argument("--out_dir", default=None, help="output dir; default data_folder/attr")
    args = parser.parse_args()

    wordmap_path = os.path.join(args.data_folder, f"WORDMAP_{args.data_name}.json")
    train_caps_path = os.path.join(args.data_folder, f"TRAIN_CAPTIONS_{args.data_name}.json")
    word_map = load_wordmap(wordmap_path)
    idx2word = {str(v): k for k, v in word_map.items()}

    with open(train_caps_path, "r") as f:
        train_captions = json.load(f)

    # infer captions_per_image from data_name (e.g., UCM_5_cap_per_img_4_min_word_freq)
    tokens = args.data_name.split("_")
    cpi = None
    for tok in tokens:
        if tok.isdigit():
            cpi = int(tok)
            break
    if cpi is None:
        raise ValueError("Could not infer captions_per_image from data_name; please set manually.")

    attr_vocab = top_k_attrs(train_captions, idx2word, args.top_k)
    labels = build_labels(train_captions, cpi, idx2word, attr_vocab)
    adj = build_pmi_adj(train_captions, idx2word, attr_vocab)

    out_dir = args.out_dir or os.path.join(args.data_folder, "attr")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "attr_vocab.json"), "w") as f:
        json.dump(attr_vocab, f)
    np.save(os.path.join(out_dir, "attr_labels.npy"), labels)
    np.save(os.path.join(out_dir, "adj.npy"), adj)
    print(f"Saved attr vocab/labels/adj to {out_dir}")


if __name__ == "__main__":
    main()
