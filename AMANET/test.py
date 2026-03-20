#!/usr/bin/env python
"""验证重新生成的数据是否一致"""
import h5py
import json
import os

print("=" * 70)
print("验证数据文件")
print("=" * 70)

# 1. 检查文件是否存在
files_to_check = [
    'data/rsicdtalk.json',
    'data/rsicdtalk_label.h5'
]

for fpath in files_to_check:
    if os.path.exists(fpath):
        size = os.path.getsize(fpath) / (1024 * 1024)  # MB
        print(f"✓ {fpath}: {size:.2f} MB")
    else:
        print(f"✗ {fpath}: 文件不存在！")

print()

# 2. 检查词表
print("=" * 70)
print("词表信息")
print("=" * 70)
with open('data/rsicdtalk.json', 'r') as f:
    data = json.load(f)

vocab_size = len(data['ix_to_word'])
print(f"词表大小 (vocab_size): {vocab_size}")
print(f"有效索引范围: 1 到 {vocab_size}")
print(f"图像数量: {len(data['images'])}")
print()

# 3. 检查标签
print("=" * 70)
print("标签数据信息")
print("=" * 70)
h5_file = h5py.File('data/rsicdtalk_label.h5', 'r')
labels = h5_file['labels'][:]

print(f"标签数组形状: {labels.shape}")
print(f"标签最小值: {labels.min()}")
print(f"标签最大值: {labels.max()}")
print()

# 4. 一致性检查
print("=" * 70)
print("一致性检查")
print("=" * 70)

if labels.max() <= vocab_size:
    print("✓ 通过: 标签范围在词表范围内")
    print(f"  词表大小: {vocab_size}")
    print(f"  最大标签: {labels.max()}")
    print()
    print("✅ 数据验证通过！可以开始训练。")
else:
    print("✗ 失败: 标签超出词表范围")
    print(f"  词表大小: {vocab_size}")
    print(f"  最大标签: {labels.max()}")
    print(f"  超出: {labels.max() - vocab_size}")
    print()
    print("⚠️  需要重新生成数据或调整配置")

h5_file.close()
