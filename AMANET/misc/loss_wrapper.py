import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt, attr_extractor=None):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.attr_extractor = attr_extractor
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag):
        out = {}
        
        # 调试：打印输入形状
        if not hasattr(self, '_debug_printed'):
            print(f"\n[DEBUG] Input shapes:")
            print(f"  fc_feats: {fc_feats.shape}")
            print(f"  att_feats: {att_feats.shape}")
            print(f"  labels: {labels.shape}")
            print(f"[DEBUG] Label statistics:")
            print(f"  labels min: {labels.min().item()}")
            print(f"  labels max: {labels.max().item()}")
            print(f"  vocab_size from opt: {self.opt.vocab_size}")
            print(f"  model vocab_size: {self.model.vocab_size if hasattr(self.model, 'vocab_size') else 'N/A'}")
            self._debug_printed = True
        
        
        # 提取属性特征（如果有属性提取器）
        attr_feats = None
        if self.attr_extractor is not None:
            with torch.no_grad():
                # att_feats: [B, num_regions, feat_dim]
                # att_masks: [B, num_regions] 或 None
                if att_masks is not None:
                    # 将 att_masks 转换为 key_padding_mask 格式（True 表示 padding）
                    key_padding_mask = (att_masks == 0)
                else:
                    key_padding_mask = None
                
                # 提取属性概率 [B, num_attributes]
                attr_feats = self.attr_extractor(att_feats, img_masks=att_masks)
        
        if not sc_flag:
            # 训练阶段：teacher forcing
            if attr_feats is not None:
                # 带属性特征的模型
                model_out = self.model(fc_feats, att_feats, labels, att_masks, attr_feats=attr_feats)
            else:
                # 标准模型
                model_out = self.model(fc_feats, att_feats, labels, att_masks)
            loss = self.crit(model_out, labels[:,1:], masks[:,1:])
        else:
            # 强化学习阶段
            self.model.eval()
            with torch.no_grad():
                if attr_feats is not None:
                    greedy_res, _ = self.model(fc_feats, att_feats, att_masks, attr_feats=attr_feats, mode='sample')
                else:
                    greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
            self.model.train()
            
            if attr_feats is not None:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, attr_feats=attr_feats, 
                                                         opt={'sample_method':'sample'}, mode='sample')
            else:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, 
                                                         opt={'sample_method':'sample'}, mode='sample')
            
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
