#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本 - 使用标注属性标签的 Caption 模型

用于训练 ShowTellAttrLabelModel，直接使用数据集中的 attribute_labels
而不是通过属性特征提取器预测属性特征。

优势：
- 属性标签是从 caption 中提取的，100% 准确
- 不需要预训练属性特征提取器
- 避免属性预测噪声影响 caption 生成质量

使用方法：
python train_with_attr_labels.py \
    --input_json data/rsicd_with_attributes_new40.json \
    --input_fc_dir data/rsicdtalk_fc \
    --input_att_dir data/rsicdtalk_att \
    --input_label_h5 data/rsicdtalk_label.h5 \
    --caption_model show_tell_attr_label \
    --attr_feat_size 40 \
    --checkpoint_path save/show_tell_attr_label \
    --id show_tell_attr_label
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle
import traceback

import opts
import models
from dataloader import *
import skimage.io
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from misc.loss_wrapper_with_attr_labels import LossWrapperWithAttrLabels
import sys
import atexit

# 多卡GPU使用单卡训练
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def eval_split_with_attr_labels(model, crit, loader, eval_kwargs={}):
    """评估函数 - 支持属性标签"""
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)

    model.eval()
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_.cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp
            
            # 获取属性标签
            attr_labels = data.get('attr_labels', None)
            if attr_labels is not None:
                attr_labels = attr_labels.cuda()

            with torch.no_grad():
                loss = crit(model(fc_feats, att_feats, labels, att_masks, attr_labels=attr_labels), 
                           labels[:,1:], masks[:,1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # 生成 caption
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        
        # 获取每个图像的属性标签（不重复）
        attr_labels = data.get('attr_labels', None)
        if attr_labels is not None:
            attr_labels = attr_labels[np.arange(loader.batch_size) * loader.seq_per_img].cuda()

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

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation performance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        from eval_utils import language_eval
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    model.train()
    return loss_sum/loss_evals, predictions, lang_stats


def train(opt):
    # 强制启用属性标签
    opt.use_attr_labels = True
    
    # Deal with feature things before anything
    opt.use_fc, opt.use_att = utils.if_use_feat(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    acc_steps = getattr(opt, 'acc_steps', 1)
    print_every = getattr(opt, 'print_every', 50)
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    log_path = os.path.join(opt.checkpoint_path, 'train_%s.txt' % opt.id)
    log_file = open(log_path, 'a', buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    atexit.register(log_file.close)
        
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    reset_training = getattr(opt, 'reset_training_state', 0) == 1
    
    if opt.start_from is not None:
        infos_path = os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')
        if os.path.isfile(infos_path):
            with open(infos_path, 'rb') as f:
                infos = utils.pickle_load(f)
                saved_model_opt = infos['opt']
                need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
                for checkme in need_be_same:
                    assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], \
                        "Command line argument and saved model disagree on '%s' " % checkme
        else:
            print("Warning: infos file not found in {}, starting with fresh infos.".format(opt.start_from))

        histories_path = os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')
        if os.path.isfile(histories_path):
            with open(histories_path, 'rb') as f:
                histories = utils.pickle_load(f)
    if len(infos) == 0 or reset_training:
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['iterators'] = loader.iterators
        infos['split_ix'] = loader.split_ix
        infos['vocab'] = loader.get_vocab()
    infos['opt'] = opt

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})
    if reset_training:
        val_result_history, loss_history, lr_history, ss_prob_history = {}, {}, {}, {}

    if not reset_training:
        loader.iterators = infos.get('iterators', loader.iterators)
        loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1 and not reset_training:
        best_val_score = infos.get('best_val_score', None)
    else:
        best_val_score = None
    
    # Early stopping variables
    patience = getattr(opt, 'early_stopping_patience', 5)
    patience_counter = 0
    best_val_score_for_early_stop = best_val_score

    opt.vocab = loader.get_vocab()
    model = models.setup(opt).cuda()
    del opt.vocab
    optimizer_suffix = getattr(opt, 'start_from_ckpt_suffix', '')
    
    dp_model = torch.nn.DataParallel(model)
    
    # 使用属性标签版本的 LossWrapper
    lw_model = LossWrapperWithAttrLabels(model, opt)
    dp_lw_model = torch.nn.DataParallel(lw_model)

    epoch_done = True
    dp_lw_model.train()

    if opt.noamopt:
        assert opt.caption_model in ['transformer','aoa'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        optimizer._step = iteration
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        
    if vars(opt).get('start_from', None) is not None and getattr(opt, 'reset_optimizer', 0) != 1:
        opt_path = os.path.join(opt.start_from, f'optimizer{optimizer_suffix}.pth')
        if not os.path.isfile(opt_path) and optimizer_suffix != '':
            print(f'Warning: {opt_path} not found, fallback to optimizer.pth')
            opt_path = os.path.join(opt.start_from, 'optimizer.pth')
        if os.path.isfile(opt_path):
            optimizer.load_state_dict(torch.load(opt_path))
            print(f'Loaded optimizer state from {opt_path}')
        else:
            print(f'Warning: optimizer state {opt_path} not found, starting with fresh optimizer')

    def save_checkpoint(model, infos, optimizer, histories=None, append=''):
        append_str = ('-' + append) if len(append) > 0 else ''
        if not os.path.isdir(opt.checkpoint_path):
            os.makedirs(opt.checkpoint_path)
        checkpoint_path = os.path.join(opt.checkpoint_path, f'model{append_str}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print("model saved to {}".format(checkpoint_path))
        optimizer_path = os.path.join(opt.checkpoint_path, f'optimizer{append_str}.pth')
        torch.save(optimizer.state_dict(), optimizer_path)
        with open(os.path.join(opt.checkpoint_path, f'infos_{opt.id}{append_str}.pkl'), 'wb') as f:
            utils.pickle_dump(infos, f)
        if histories:
            with open(os.path.join(opt.checkpoint_path, f'histories_{opt.id}{append_str}.pkl'), 'wb') as f:
                utils.pickle_dump(histories, f)
        # 额外保存带分数的 best ckpt，避免后续 run 覆盖
        if append == 'best' and 'best_val_score' in infos:
            score = infos['best_val_score']
            score_tag = f"{score:.4f}".replace('.', '_')
            extra_name = os.path.join(opt.checkpoint_path, f'model-best-{score_tag}.pth')
            torch.save(model.state_dict(), extra_name)
            print(f"model saved to {extra_name}")
            with open(os.path.join(opt.checkpoint_path, f'infos_{opt.id}-best-{score_tag}.pkl'), 'wb') as f:
                utils.pickle_dump(infos, f)

    try:
        while True:
            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr)
                    
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    sc_flag = False

                epoch_done = False
                print("Epoch {}/{}".format(epoch+1, opt.max_epochs))
            
            start = time.time()
            if (opt.use_warmup == 1) and (iteration < opt.noamopt_warmup):
                opt.current_lr = opt.learning_rate * (iteration+1) / opt.noamopt_warmup
                utils.set_lr(optimizer, opt.current_lr)
            data = loader.get_batch('train')
            read_time = time.time() - start
            if iteration % print_every == 0:
                print('Read data:', read_time)

            if (iteration % acc_steps == 0):
                optimizer.zero_grad()
            
            torch.cuda.synchronize()
            start = time.time()
            
            # 准备数据，包括属性标签
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp
            
            # 获取属性标签
            attr_labels = data.get('attr_labels', None)
            if attr_labels is not None:
                attr_labels = attr_labels.cuda()

            # 前向传播，传入属性标签
            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, 
                                    data['gts'], torch.arange(0, len(data['gts'])), 
                                    sc_flag, attr_labels=attr_labels)

            loss = model_out['loss'].mean()
            loss_sp = loss / acc_steps

            loss_sp.backward()
            if ((iteration+1) % acc_steps == 0):
                utils.clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
            torch.cuda.synchronize()
            train_loss = loss.item()
            end = time.time()
            
            if iteration % print_every == 0:
                if not sc_flag:
                    print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(iteration, epoch, train_loss, end - start))
                else:
                    print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                        .format(iteration, epoch, model_out['reward'].mean(), end - start))

            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            if (iteration % opt.losses_log_every == 0):
                add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    add_summary_value(tb_summary_writer, 'avg_reward', model_out['reward'].mean(), iteration)

                loss_history[iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                lr_history[iteration] = opt.current_lr
                ss_prob_history[iteration] = model.ss_prob

            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            
            if (iteration % opt.save_checkpoint_every == 0):
                eval_kwargs = {'split': 'val', 'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                
                # 使用支持属性标签的评估函数
                val_loss, predictions, lang_stats = eval_split_with_attr_labels(
                    dp_model, lw_model.crit, loader, eval_kwargs)

                if opt.reduce_on_plateau:
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        optimizer.scheduler_step(val_loss)
                        
                add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        add_summary_value(tb_summary_writer, k, v, iteration)
                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss

                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Early stopping logic
                if best_val_score_for_early_stop is None or current_score > best_val_score_for_early_stop:
                    best_val_score_for_early_stop = current_score
                    patience_counter = 0
                    print(f"New best validation score: {current_score:.4f}")
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} evaluations. Best: {best_val_score_for_early_stop:.4f}, Current: {current_score:.4f}")
                    
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {patience} evaluations without improvement")
                        print(f"Best validation score was: {best_val_score_for_early_stop:.4f}")
                        save_checkpoint(model, infos, optimizer, histories, append='early_stop')
                        break

                infos['best_val_score'] = best_val_score
                infos['patience_counter'] = patience_counter
                infos['best_val_score_for_early_stop'] = best_val_score_for_early_stop
                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history

                save_checkpoint(model, infos, optimizer, histories)
                if getattr(opt, 'save_history_ckpt', 0):
                    save_checkpoint(model, infos, optimizer, append=str(iteration))

                if best_flag:
                    save_checkpoint(model, infos, optimizer, append='best')

            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break
                
    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        save_checkpoint(model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


if __name__ == '__main__':
    opt = opts.parse_opt()
    train(opt)
