#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估脚本 - 使用标注属性标签的 Caption 模型

用于评估 ShowTellAttrLabelModel

使用方法：
python eval_with_attr_labels.py \
    --model save/show_tell_attr_label/model-best.pth \
    --infos_path save/show_tell_attr_label/infos_show_tell_attr_label-best.pkl \
    --input_json data/rsicd_with_attributes_new40.json \
    --input_fc_dir data/rsicdtalk_fc \
    --input_att_dir data/rsicdtalk_att \
    --input_label_h5 data/rsicdtalk_label.h5 \
    --beam_size 3 \
    --language_eval 1 \
    --split test
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import argparse
import os

import torch
import numpy as np

import misc.utils as utils
from dataloader import DataLoader
import models

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def _get_annotation_file(dataset, split):
    dataset_str = (dataset or '').lower()
    if 'sydney' in dataset_str:
        prefix = 'coco-caption/annotations/captions_sydney_'
    elif 'ucm' in dataset_str:
        prefix = 'coco-caption/annotations/captions_ucm_'
    elif 'rsicd' in dataset_str:
        prefix = 'coco-caption/annotations/captions_rsicd_'
    else:
        prefix = 'coco-caption/annotations/captions_'
    if split in ['val', 'train']:
        return prefix + 'val.json'
    elif split == 'test':
        return prefix + 'test.json'
    else:
        raise ValueError("split must be train/val/test")

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    try:
        annFile = _get_annotation_file(dataset, split)
    except ValueError:
        print("ERROR: split is not right!")
        return None
        
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()
    
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    
    json.dump(preds_filt, open(cache_path, 'w'))
    
    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    
    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def eval_split_with_attr_labels(model, loader, eval_kwargs={}):
    """评估函数 - 支持属性标签"""
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    num_images = eval_kwargs.get('num_images', -1)
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)
    device = eval_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    model.eval()
    loader.reset_iterator(split)

    n = 0
    predictions = []
    
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        # 生成 caption
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [_.to(device) if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        
        # 获取每个图像的属性标签
        attr_labels = data.get('attr_labels', None)
        if attr_labels is not None:
            attr_labels = attr_labels[np.arange(loader.batch_size) * loader.seq_per_img].to(device)

        with torch.no_grad():
            seq = model(fc_feats, att_feats, att_masks, attr_labels=attr_labels, 
                       opt=eval_kwargs, mode='sample')[0].data

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                actual_model = model.module if hasattr(model, 'module') else model
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] 
                                for _ in actual_model.done_beams[i]]))
                print('--' * 10)
                
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            
            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating... %d/%d' % (ix0 - 1, ix1))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)
        print('\n=== Evaluation Results ===')
        for k, v in lang_stats.items():
            print(f'{k}: {v:.4f}')

    return predictions, lang_stats


def main():
    parser = argparse.ArgumentParser()
    
    # 模型路径
    parser.add_argument('--model', type=str, required=True,
                        help='path to model checkpoint')
    parser.add_argument('--infos_path', type=str, required=True,
                        help='path to infos pkl file')
    
    # 数据路径
    parser.add_argument('--input_json', type=str, default='data/rsicd_with_attributes_new40.json',
                        help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='data/rsicdtalk_fc',
                        help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='data/rsicdtalk_att',
                        help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_box_dir', type=str, default='0',
                        help='path to the directory containing the boxes')
    parser.add_argument('--input_label_h5', type=str, default='data/rsicdtalk_label.h5',
                        help='path to the h5file containing the preprocessed dataset')
    
    # 评估参数
    parser.add_argument('--split', type=str, default='test',
                        help='which split to evaluate: val|test')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for evaluation')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='beam size for beam search')
    parser.add_argument('--language_eval', type=int, default=1,
                        help='evaluate language metrics (1=yes, 0=no)')
    parser.add_argument('--num_images', type=int, default=-1,
                        help='how many images to evaluate (-1 = all)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='print progress')
    parser.add_argument('--verbose_beam', type=int, default=0,
                        help='print beam search results')
    parser.add_argument('--dump_path', type=int, default=0,
                        help='dump file path in predictions')
    parser.add_argument('--dump_json', type=int, default=1,
                        help='dump json with predictions')
    parser.add_argument('--id', type=str, default='eval',
                        help='id for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run evaluation on (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='dataloader worker count (use 0 for restricted environments)')
    
    args = parser.parse_args()
    
    # 加载模型信息
    with open(args.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)
    
    # 获取保存的 opt
    opt = infos['opt']
    
    # 更新路径参数
    opt.input_json = args.input_json
    opt.input_fc_dir = args.input_fc_dir
    opt.input_att_dir = args.input_att_dir
    opt.input_box_dir = args.input_box_dir
    opt.input_label_h5 = args.input_label_h5
    opt.batch_size = args.batch_size
    opt.num_workers = args.num_workers
    
    # 启用属性标签
    opt.use_attr_labels = True
    
    # 创建数据加载器
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    
    # 加载词汇表
    opt.vocab = loader.get_vocab()
    
    # 选择设备
    force_cpu = args.device == 'cpu' or not torch.cuda.is_available()
    if force_cpu:
        device = torch.device('cpu')
        # 在无 GPU 环境下将 .cuda() 调用变为 no-op，避免下游模型报错
        def _noop_cuda(self, device=None):
            return self
        torch.Tensor.cuda = _noop_cuda  # type: ignore
        torch.nn.Module.cuda = _noop_cuda  # type: ignore
    else:
        device = torch.device(args.device)
    opt.device = device.type

    # 创建模型
    model = models.setup(opt)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    del opt.vocab
    
    # 评估参数
    eval_kwargs = {
        'split': args.split,
        'dataset': args.input_json,
        'beam_size': args.beam_size,
        'language_eval': args.language_eval,
        'num_images': args.num_images,
        'verbose': args.verbose,
        'verbose_beam': args.verbose_beam,
        'dump_path': args.dump_path,
        'id': args.id,
        'device': device,
    }
    
    # 评估
    predictions, lang_stats = eval_split_with_attr_labels(model, loader, eval_kwargs)
    
    # 保存预测结果
    if args.dump_json:
        if not os.path.isdir('eval_results'):
            os.makedirs('eval_results')
        json_path = os.path.join('eval_results', f'{args.id}_{args.split}_predictions.json')
        with open(json_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f'Predictions saved to {json_path}')


if __name__ == '__main__':
    main()
