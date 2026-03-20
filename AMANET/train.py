#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from misc.loss_wrapper import LossWrapper
from models.AttributeFeatureExtractor import AttributeFeatureExtractor
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

def train(opt):
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
    # opt.start_from = None
    if opt.start_from is not None:
        infos_path = os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')
        if os.path.isfile(infos_path):
            with open(infos_path, 'rb') as f:
                infos = utils.pickle_load(f)
                saved_model_opt = infos['opt']
                need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
                for checkme in need_be_same:
                    assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        else:
            print("Warning: infos file not found in {}, starting with fresh infos.".format(opt.start_from))

        hist_path = os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')
        if os.path.isfile(hist_path):
            with open(hist_path, 'rb') as f:
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
    dp_model = torch.nn.DataParallel(model)
    optimizer_suffix = getattr(opt, 'start_from_ckpt_suffix', '')
    
    # 加载属性提取器（仅当使用带属性特征的 caption 模型时）
    attr_extractor = None
    if opt.caption_model in ['show_tell_attr', 'show_tell_attr_all', 'show_tell_attr_attn',
                             'show_tell_attr_only', 'show_tell_attr_init'] and opt.use_attr_feats:
        print(f"Loading attribute extractor from {opt.attr_extractor_path}")
        attr_ckpt = torch.load(opt.attr_extractor_path, map_location='cuda')
        attr_args = attr_ckpt.get('args', {})
        attr_extractor = AttributeFeatureExtractor(
            feat_dim=attr_args.get('feat_dim', opt.att_feat_size),
            num_attributes=attr_ckpt.get('num_attributes', opt.attr_feat_size),
            d_model=attr_args.get('d_model', 512),
            nhead=attr_args.get('nhead', 8),
            num_layers=attr_args.get('num_layers', 6),
            dim_feedforward=attr_args.get('dim_feedforward', 2048),
            dropout=attr_args.get('dropout', 0.1),
        ).cuda()
        attr_extractor.load_state_dict(attr_ckpt['model_state'], strict=False)
        attr_extractor.eval()  # 属性提取器始终保持评估模式
        print(f"Attribute extractor loaded successfully (num_attributes={attr_extractor.num_attributes})")
    
    lw_model = LossWrapper(model, opt, attr_extractor=attr_extractor)
    dp_lw_model = torch.nn.DataParallel(lw_model)

    epoch_done = True
    # Assure in training mode
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
    # Load the optimizer
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
        if len(append) > 0:
            append = '-' + append
        # if checkpoint_path doesn't exist
        if not os.path.isdir(opt.checkpoint_path):
            os.makedirs(opt.checkpoint_path)
        checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' %(append))
        torch.save(model.state_dict(), checkpoint_path)
        print("model saved to {}".format(checkpoint_path))
        optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
        torch.save(optimizer.state_dict(), optimizer_path)
        with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            utils.pickle_dump(infos, f)
        if histories:
            with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
                utils.pickle_dump(histories, f)

    try:
        while True:
            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate  ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                # If start self critical training
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
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'], torch.arange(0, len(data['gts'])), sc_flag)

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

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary
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

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            
            # make evaluation on validation set, and save model
            if (iteration % opt.save_checkpoint_every == 0):
                # eval model
                eval_kwargs = {'split': 'val',
                                'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(
                    dp_model, lw_model.crit, loader, eval_kwargs)

                if opt.reduce_on_plateau:
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        optimizer.scheduler_step(val_loss)
                # Write validation result into summary
                add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        add_summary_value(tb_summary_writer, k, v, iteration)
                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if is improving on validation result
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
                        # Save final checkpoint before stopping
                        save_checkpoint(model, infos, optimizer, histories, append='early_stop')
                        break

                # Dump miscalleous informations
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

            # Stop if reaching max epochs
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
