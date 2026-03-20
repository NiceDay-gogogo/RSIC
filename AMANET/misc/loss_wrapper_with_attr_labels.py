"""
LossWrapper - 支持从数据集传递属性标签

用于 ShowTellAttrLabelModel，直接使用标注的属性标签而不是预测的属性特征
"""
import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward


class LossWrapperWithAttrLabels(torch.nn.Module):
    """支持属性标签的 LossWrapper"""
    
    def __init__(self, model, opt):
        super(LossWrapperWithAttrLabels, self).__init__()
        self.opt = opt
        self.model = model
        
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        self.attr_crit = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.attr_loss_weight = getattr(opt, 'attr_loss_weight', 1.0)
        self.cap_loss_weight = getattr(opt, 'cap_loss_weight', 1.0)

    def _maybe_compute_attr_loss(self, attr_labels):
        if attr_labels is None:
            return None
        if not getattr(self.model, 'use_internal_attr', False):
            return None
        getter = getattr(self.model, 'get_attr_logits', None)
        if getter is None:
            return None
        attr_logits = getter()
        if attr_logits is None:
            return None
        return self.attr_crit(attr_logits, attr_labels)

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, attr_labels=None):
        """
        Args:
            fc_feats: [B, fc_feat_size] 全局图像特征
            att_feats: [B, num_regions, att_feat_size] 区域特征
            labels: [B, seq_len] 目标序列
            masks: [B, seq_len] 序列mask
            att_masks: [B, num_regions] 区域mask
            gts: ground truth captions
            gt_indices: gt索引
            sc_flag: 是否使用self-critical训练
            attr_labels: [B, attr_feat_size] 属性标签（0/1向量）
        """
        out = {}
        
        # 调试：打印输入形状
        if not hasattr(self, '_debug_printed'):
            print(f"\n[DEBUG LossWrapperWithAttrLabels] Input shapes:")
            print(f"  fc_feats: {fc_feats.shape}")
            print(f"  att_feats: {att_feats.shape}")
            print(f"  labels: {labels.shape}")
            if attr_labels is not None:
                print(f"  attr_labels: {attr_labels.shape}")
            else:
                print(f"  attr_labels: None")
            self._debug_printed = True
        attr_labels_device = None
        if attr_labels is not None:
            attr_labels_device = attr_labels.to(fc_feats.device).float()

        model_attr_input = attr_labels_device
        if getattr(self.model, 'use_internal_attr', False):
            model_attr_input = None

        attr_loss_tensor = None

        if not sc_flag:
            # 训练阶段：teacher forcing
            # 使用 attr_labels 参数传递属性标签
            model_out = self.model(fc_feats, att_feats, labels, att_masks, attr_labels=model_attr_input)
            cap_loss = self.crit(model_out, labels[:,1:], masks[:,1:])
            attr_loss_tensor = self._maybe_compute_attr_loss(attr_labels_device)
            loss = self.cap_loss_weight * cap_loss
            if attr_loss_tensor is not None:
                loss = loss + self.attr_loss_weight * attr_loss_tensor
            out['cap_loss'] = cap_loss.detach()
            if attr_loss_tensor is not None:
                out['attr_loss'] = attr_loss_tensor.detach()
        else:
            # 强化学习阶段
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, attr_labels=model_attr_input, mode='sample')
            self.model.train()
            
            gen_result, sample_logprobs = self.model(
                fc_feats, att_feats, att_masks, attr_labels=model_attr_input,
                opt={'sample_method':'sample'}, mode='sample'
            )
            
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            cap_loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            attr_loss_tensor = self._maybe_compute_attr_loss(attr_labels_device)
            loss = self.cap_loss_weight * cap_loss
            if attr_loss_tensor is not None:
                loss = loss + self.attr_loss_weight * attr_loss_tensor
                out['attr_loss'] = attr_loss_tensor.detach()
            out['reward'] = reward[:,0].mean()
            out['cap_loss'] = cap_loss.detach()
            
        out['loss'] = loss
        return out
