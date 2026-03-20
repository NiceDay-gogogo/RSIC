"""
端到端训练脚本

直接从原始图像训练 Image Captioning 模型
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
import torch.optim as optim
import numpy as np
import time
import os
import sys
import atexit

import opts
import models
from dataloader_images import ImageDataLoader
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
# 不使用 LossWrapper，直接计算损失

# 日志输出
class Tee:
    def __init__(self, fname, mode='a'):
        self.file = open(fname, mode)
        self.stdout = sys.stdout
        sys.stdout = self
        
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
        
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()


def train(opt):
    # 创建输出目录
    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    
    # 日志文件
    log_file = os.path.join(opt.checkpoint_path, f'train_{opt.id}.txt')
    tee = Tee(log_file, 'a')
    atexit.register(lambda: tee.file.close())
    
    # 数据加载器
    loader = ImageDataLoader(opt)
    opt.vocab_size = loader.get_vocab_size()
    opt.seq_length = loader.get_seq_length()
    opt.vocab = loader.get_vocab()
    
    # 模型
    model = models.setup(opt)
    model = model.cuda()
    
    # 损失函数
    if opt.label_smoothing > 0:
        crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
    else:
        crit = utils.LanguageModelCriterion()
    
    # DataParallel
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        dp_model = nn.DataParallel(model)
    else:
        dp_model = model
    
    # 优化器
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.learning_rate,
        weight_decay=opt.weight_decay
    )
    
    # 学习率调度器
    if opt.reduce_on_plateau:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', factor=0.5, patience=3, verbose=True
        )
    
    # 加载检查点
    iteration = 0
    epoch = 0
    best_val_score = None
    no_improve_count = 0
    
    if vars(opt).get('start_from', None) is not None:
        model_path = os.path.join(opt.start_from, 'model.pth')
        optimizer_path = os.path.join(opt.start_from, 'optimizer.pth')
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f'Loaded model from {model_path}')
        
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
            print(f'Loaded optimizer from {optimizer_path}')
        
        infos_path = os.path.join(opt.start_from, f'infos_{opt.id}.pkl')
        if os.path.exists(infos_path):
            import pickle
            with open(infos_path, 'rb') as f:
                infos = pickle.load(f)
                iteration = infos.get('iter', 0)
                epoch = infos.get('epoch', 0)
                best_val_score = infos.get('best_val_score', None)
    
    # Self-critical 训练
    sc_flag = False
    if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
        sc_flag = True
        init_scorer(opt.cached_tokens)
    
    # 开始训练
    print(f'\nStarting training from epoch {epoch}, iteration {iteration}')
    print(f'Model: {opt.caption_model}')
    print(f'Backbone: {getattr(opt, "backbone", "resnet101")}')
    print(f'Fine-tune CNN: {getattr(opt, "fine_tune_cnn", True)}')
    
    while True:
        # 学习率衰减
        if opt.learning_rate_decay_start >= 0 and epoch >= opt.learning_rate_decay_start:
            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            decay_factor = opt.learning_rate_decay_rate ** frac
            opt.current_lr = opt.learning_rate * decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.current_lr
        else:
            opt.current_lr = opt.learning_rate
        
        # Scheduled Sampling
        if opt.scheduled_sampling_start >= 0 and epoch >= opt.scheduled_sampling_start:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.scheduled_sampling_increase_every * frac, opt.scheduled_sampling_max_prob)
            model.ss_prob = opt.ss_prob
        
        # Self-critical 开关
        if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
            if not sc_flag:
                sc_flag = True
                init_scorer(opt.cached_tokens)
        
        print(f'\nEpoch {epoch + 1}/{opt.max_epochs}')
        
        # 训练一个 epoch
        model.train()
        start = time.time()
        
        data = loader.get_batch('train')
        while True:
            torch.cuda.synchronize()
            
            # 准备数据
            images = data['images'].cuda()
            labels = torch.LongTensor(data['labels']).cuda()
            masks = torch.FloatTensor(data['masks']).cuda()
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向传播
            if not sc_flag:
                # Cross-entropy 训练
                # 对于端到端模型，直接传入图像
                model_out = dp_model(images, None, labels, None)
                loss = crit(model_out, labels[:, 1:], masks[:, 1:])
            else:
                # Self-critical 训练
                model.eval()
                with torch.no_grad():
                    greedy_res, _ = model(images, None, mode='sample')
                model.train()
                gen_result, sample_logprobs = model(images, None, mode='sample', opt={'sample_method': 'sample'})
                gts = data['gts']
                reward = get_self_critical_reward(greedy_res, gts, gen_result, opt)
                reward = torch.from_numpy(reward).float().cuda()
                loss = -(sample_logprobs * reward).mean()
            
            # 反向传播
            loss.backward()
            clip_grad_value_(model.parameters(), opt.grad_clip)
            optimizer.step()
            
            train_loss = loss.item()
            end = time.time()
            
            # 打印日志
            if iteration % opt.print_every == 0:
                print(f'iter {iteration} (epoch {epoch}), train_loss = {train_loss:.3f}, time/batch = {end - start:.3f}')
            
            # 更新迭代器
            iteration += 1
            start = time.time()
            
            # 获取下一个 batch
            data = loader.get_batch('train')
            
            # epoch 结束
            if data['bounds']['wrapped']:
                break
            
            # 保存和评估
            if iteration % opt.save_checkpoint_every == 0:
                # 评估
                val_loss, predictions, lang_stats = eval_split(model, loader, 'val', opt)
                
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = -val_loss
                
                # 学习率调度
                if opt.reduce_on_plateau:
                    scheduler.step(current_score)
                
                # 保存检查点
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    no_improve_count = 0
                    
                    print(f'New best validation score: {best_val_score:.4f}')
                    
                    # 保存最佳模型
                    torch.save(model.state_dict(), os.path.join(opt.checkpoint_path, 'model.pth'))
                    torch.save(model.state_dict(), os.path.join(opt.checkpoint_path, 'model-best.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(opt.checkpoint_path, 'optimizer.pth'))
                    
                    # 保存 infos
                    import pickle
                    infos = {
                        'iter': iteration,
                        'epoch': epoch,
                        'best_val_score': best_val_score,
                        'opt': opt
                    }
                    with open(os.path.join(opt.checkpoint_path, f'infos_{opt.id}.pkl'), 'wb') as f:
                        pickle.dump(infos, f)
                else:
                    no_improve_count += 1
                    print(f'No improvement for {no_improve_count} evaluations. Best: {best_val_score:.4f}, Current: {current_score:.4f}')
                
                # Early stopping
                if no_improve_count >= opt.early_stopping_patience:
                    print(f'Early stopping triggered after {no_improve_count} evaluations without improvement.')
                    torch.save(model.state_dict(), os.path.join(opt.checkpoint_path, 'model-early_stop.pth'))
                    return
                
                model.train()
        
        epoch += 1
        
        if epoch >= opt.max_epochs:
            print(f'Training completed. Best validation score: {best_val_score:.4f}')
            break


def eval_split(model, loader, split, opt):
    """评估函数"""
    model.eval()
    
    n = 0
    loss_sum = 0
    predictions = []
    
    loader.reset_iterator(split)
    
    with torch.no_grad():
        while True:
            data = loader.get_batch(split)
            
            images = data['images'].cuda()
            
            # 生成句子
            seq, _ = model(images, None, mode='sample')
            
            # 收集预测
            seq_per_img = loader.seq_per_img
            for k in range(len(data['infos'])):
                entry = {
                    'image_id': data['infos'][k]['id'],
                    'caption': utils.decode_sequence(loader.get_vocab(), seq[k*seq_per_img:k*seq_per_img+1])[0]
                }
                predictions.append(entry)
            
            n += len(data['infos'])
            
            print(f'evaluating {split} performance... {n}/{len(loader.split_ix[split])}')
            
            if data['bounds']['wrapped']:
                break
            
            if opt.val_images_use > 0 and n >= opt.val_images_use:
                break
    
    # 语言评估
    lang_stats = {}
    if opt.language_eval == 1:
        from eval_utils import language_eval
        lang_stats = language_eval(
            opt.input_json,
            predictions,
            opt.id,
            split
        )
        
        # 打印评估结果
        print('Validation Results:')
        for k, v in lang_stats.items():
            print(f'  {k}: {v:.4f}')
    
    return loss_sum / max(n, 1), predictions, lang_stats


if __name__ == '__main__':
    opt = opts.parse_opt()
    
    # 添加端到端特有的参数默认值
    if not hasattr(opt, 'image_dir'):
        opt.image_dir = 'data/RSICD/images'
    if not hasattr(opt, 'image_size'):
        opt.image_size = 224
    
    train(opt)
