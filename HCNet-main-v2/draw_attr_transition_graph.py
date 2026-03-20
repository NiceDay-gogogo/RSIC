import argparse
import json
import math
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx


# 基本停用词（你也可以直接复制 gen_attr_gcn.py 里的 STOPWORDS 集合过来更完整）
STOPWORDS = {
    "a","an","the","and","or","of","in","on","at","to","with","by","is","are","was","were",
    "be","been","being","this","that","there","it","as","for","from","into","over","under",
    "no","not","one","two","three","four","five","six","seven","eight","nine","ten",
    "some","any","each","other","others","both","all","many","few","more","most","much",
    "very","has","have","had","can","could","should","would","will","may","might",
    "them","they","their","its","those","these","which","what","who",
    "beside","through","across","between","around","while","without","next",
}

TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(sentence: str):
    words = TOKEN_RE.findall(sentence.lower())
    # 简单清洗：去停用词、长度<=1
    words = [w for w in words if len(w) > 1 and w not in STOPWORDS]
    return words


def build_attr_vocab(sentences, top_k=200):
    cnt = Counter()
    for s in sentences:
        for w in tokenize(s):
            cnt[w] += 1
    vocab = [w for w, _ in cnt.most_common(top_k)]
    return vocab, cnt


def extract_attr_sequence(sentence, vocab_set):
    # 保留原句顺序，只取在属性词表里的token
    tokens = tokenize(sentence)
    return [w for w in tokens if w in vocab_set]


def build_transition(sentences, vocab):
    vocab_set = set(vocab)
    fre = Counter()
    out_total = Counter()

    for s in sentences:
        seq = extract_attr_sequence(s, vocab_set)
        for a, b in zip(seq, seq[1:]):
            fre[(a, b)] += 1
            out_total[a] += 1

    # 行归一化转移概率 U(a->b) = fre(a,b)/sum_b fre(a,b)
    U = {}
    for (a, b), c in fre.items():
        U[(a, b)] = c / out_total[a] if out_total[a] > 0 else 0.0
    return U, fre, out_total


def draw_graph(vocab, freq_cnt, U, out_png,
               top_nodes=120, max_edges=800, min_w=0.02, seed=7):
    # 选节点：按出现频次取前 top_nodes
    nodes = [w for w, _ in freq_cnt.most_common(top_nodes)]
    node_set = set(nodes)

    # 选边：只保留两端都在 node_set 且权重大于阈值
    edges = [((a, b), w) for (a, b), w in U.items() if a in node_set and b in node_set and w >= min_w]
    edges.sort(key=lambda x: x[1], reverse=True)
    edges = edges[:max_edges]

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n, freq=freq_cnt[n])
    for (a, b), w in edges:
        G.add_edge(a, b, weight=w)

    # 布局（spring_layout 适合这种“团状”效果）
    pos = nx.spring_layout(G, seed=seed, k=0.35)

    # 节点大小：按词频缩放
    freqs = [G.nodes[n]["freq"] for n in G.nodes]
    fmin, fmax = min(freqs), max(freqs)
    def scale_node(x):
        if fmax == fmin:
            return 300
        return 150 + 1200 * (x - fmin) / (fmax - fmin)

    node_sizes = [scale_node(G.nodes[n]["freq"]) for n in G.nodes]

    # 边宽：按权重缩放
    wts = [G.edges[e]["weight"] for e in G.edges]
    if wts:
        wmin, wmax = min(wts), max(wts)
    else:
        wmin = wmax = 0.0

    def scale_edge(w):
        if wmax == wmin:
            return 0.6
        return 0.2 + 3.5 * (w - wmin) / (wmax - wmin)

    edge_widths = [scale_edge(G.edges[e]["weight"]) for e in G.edges]

    plt.figure(figsize=(12, 12), dpi=220)
    ax = plt.gca()
    ax.set_axis_off()

    # 画边（红色、半透明）
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.20,
        edge_color="#d94b5a",
        arrows=False,  # 想画箭头可以改 True，但会更乱
    )

    # 画点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color="#f5c6cb",
        edgecolors="#a33",
        linewidths=0.6,
        alpha=0.95
    )

    # 画标签（只给最频繁的前 N_label 个打标签，避免太乱）
    N_label = min(40, len(nodes))
    label_nodes = [w for w, _ in freq_cnt.most_common(N_label) if w in node_set]
    labels = {n: n for n in label_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color="black")

    plt.title("Attribute Transition Graph (nodes=attributes, edges=U(a->b))", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved: {out_png}")
    print(f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, min_w={min_w}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_json", required=True)
    ap.add_argument("--out_png", default="attr_graph.png")
    ap.add_argument("--top_k_vocab", type=int, default=300)
    ap.add_argument("--top_nodes", type=int, default=120)
    ap.add_argument("--max_edges", type=int, default=800)
    ap.add_argument("--min_w", type=float, default=0.02)
    args = ap.parse_args()

    data = json.loads(open(args.dataset_json, "r", encoding="utf-8").read())
    # 只用训练集句子（如 split 字段存在）
    sentences = []
    for im in data["images"]:
        if im.get("split") not in (None, "train"):
            continue
        for s in im["sentences"]:
            sentences.append(s["raw"])

    vocab, freq_cnt = build_attr_vocab(sentences, top_k=args.top_k_vocab)
    U, fre, out_total = build_transition(sentences, vocab)

    draw_graph(
        vocab=vocab,
        freq_cnt=freq_cnt,
        U=U,
        out_png=args.out_png,
        top_nodes=args.top_nodes,
        max_edges=args.max_edges,
        min_w=args.min_w,
        seed=7,
    )


if __name__ == "__main__":
    main()
