#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 attribute_labels 从原始标注文件添加到 rsicdtalk.json 中

这样可以保持 rsicdtalk.json 的格式和 h5 文件的索引对应关系不变
"""
import json

# 文件路径
attr_json = 'data/rsicd_with_attributes_new40.json'  # 原始标注（含 attribute_labels）
talk_json = 'data/rsicdtalk.json'                     # 预处理后的文件
output_json = 'data/rsicdtalk_attr.json'              # 新文件，和原来的区分开

print(f'Loading {attr_json}...')
with open(attr_json, 'r') as f:
    attr_data = json.load(f)

print(f'Loading {talk_json}...')
with open(talk_json, 'r') as f:
    talk_data = json.load(f)

# 建立 filename -> attribute_labels 的映射
# 原始文件用 filename 字段
filename_to_attrs = {}
for img in attr_data['images']:
    filename = img.get('filename', '')
    attr_labels = img.get('attribute_labels', None)
    if filename and attr_labels is not None:
        filename_to_attrs[filename] = attr_labels

print(f'Found {len(filename_to_attrs)} images with attribute_labels in {attr_json}')

# 给 rsicdtalk.json 中的每张图添加 attribute_labels
# rsicdtalk.json 用 file_path 字段（格式如 "airport_1.jpg"）
matched = 0
for img in talk_data['images']:
    file_path = img.get('file_path', '')
    
    if file_path in filename_to_attrs:
        img['attribute_labels'] = filename_to_attrs[file_path]
        matched += 1
    else:
        print(f'Warning: No attribute_labels found for {file_path}')
        img['attribute_labels'] = [0] * 40

print(f'Matched {matched}/{len(talk_data["images"])} images')

# 保存
print(f'Saving to {output_json}...')
with open(output_json, 'w') as f:
    json.dump(talk_data, f)

print('Done!')
print(f'\n现在可以直接使用 --input_json {output_json} 训练了')
