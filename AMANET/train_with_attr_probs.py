#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本 - 使用概率属性向量的 Caption 模型

流程与 train_with_attr_labels.py 完全一致，只是默认使用
caption_model=show_tell_attr_prob，并假定输入 json 中的
attribute_labels 已经被概率分布（而非 0/1）覆盖。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import opts
from train_with_attr_labels import train as train_with_attr_labels


if __name__ == "__main__":
    opt = opts.parse_opt()
    opt.caption_model = 'show_tell_attr_prob'
    if not hasattr(opt, 'attr_feat_size') or opt.attr_feat_size is None:
        opt.attr_feat_size = 40
    train_with_attr_labels(opt)
