import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EnhancedFocalLoss(nn.Module):
    """增强版Focal Loss，支持自适应权重调整"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', adaptive_gamma=True):
        super().__init__()
        self.alpha = alpha
        self.base_gamma = gamma
        self.adaptive_gamma = adaptive_gamma
        self.reduction = reduction
        # 动态gamma调整
        self.gamma_scheduler = nn.Parameter(torch.tensor(gamma))
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 动态调整gamma
        if self.adaptive_gamma:
            # 基于困难样本比例调整gamma
            with torch.no_grad():
                p_t = probs * targets + (1 - probs) * (1 - targets)
                hard_ratio = (p_t < 0.5).float().mean()
                adaptive_gamma = self.base_gamma + hard_ratio * 2.0
            focal_weight = (1 - p_t) ** adaptive_gamma
        else:
            focal_weight = (1 - p_t) ** self.base_gamma
        
        focal_loss = focal_weight * bce_loss
        
        # 类别权重调整
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.to(logits.device)
                alpha_t = alpha_t * targets + (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class EnhancedAttributeTransformerLayer(nn.Module):
    """增强的Transformer层，添加残差连接和门控机制"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, use_relative_pos=True):
        super().__init__()
        self.d_model = d_model
        
        # 增强的交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # 门控残差连接
        self.attn_gate = nn.Parameter(torch.tensor(1.0))
        self.norm1 = nn.LayerNorm(d_model)
        
        # 增强的FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_gate = nn.Parameter(torch.tensor(1.0))
        self.norm2 = nn.LayerNorm(d_model)
        
        # 相对位置编码
        if use_relative_pos:
            self.relative_pos_encoding = RelativePositionEncoding(d_model, max_len=100)
        else:
            self.relative_pos_encoding = None
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, attr_tokens, img_feats, key_padding_mask=None, pos_encoding=None):
        # 残差连接1: 注意力层
        residual = attr_tokens
        
        # 添加相对位置编码
        if self.relative_pos_encoding is not None:
            pos_bias = self.relative_pos_encoding(
                attr_tokens.size(1), img_feats.size(1), device=attr_tokens.device
            )
            attn_output, attn_weights = self.cross_attn(
                query=attr_tokens + pos_encoding if pos_encoding is not None else attr_tokens,
                key=img_feats,
                value=img_feats,
                key_padding_mask=key_padding_mask,
                attn_mask=pos_bias.to(attr_tokens.device, dtype=attr_tokens.dtype) if pos_bias is not None else None
            )
        else:
            attn_output, attn_weights = self.cross_attn(
                query=attr_tokens,
                key=img_feats,
                value=img_feats,
                key_padding_mask=key_padding_mask
            )
        
        # 门控残差连接
        attr_tokens = self.norm1(residual + self.attn_gate * self.dropout(attn_output))
        
        # 残差连接2: FFN层
        residual = attr_tokens
        ffn_output = self.ffn(attr_tokens)
        attr_tokens = self.norm2(residual + self.ffn_gate * ffn_output)
        
        return attr_tokens, attn_weights


class RelativePositionEncoding(nn.Module):
    """相对位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(2 * max_len + 1, d_model)
        self.proj = nn.Linear(d_model, 1, bias=False)
        
    def forward(self, seq_len_q, seq_len_k, device=None):
        if device is None:
            device = self.embedding.weight.device
        range_vec_q = torch.arange(seq_len_q, device=device)
        range_vec_k = torch.arange(seq_len_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_len, self.max_len)
        final_mat = distance_mat_clipped + self.max_len
        rel_emb = self.embedding(final_mat)  # [seq_len_q, seq_len_k, d_model]
        bias = self.proj(rel_emb).squeeze(-1)  # [seq_len_q, seq_len_k]
        return bias


class FeatureEnhancementModule(nn.Module):
    """特征增强模块，提升输入特征质量"""
    
    def __init__(self, feat_dim, d_model, enhancement_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        
        # 特征增强层
        self.enhance_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.LayerNorm(d_model)
            ) for _ in range(enhancement_layers)
        ])
        
        # 特征重要性权重
        self.feature_importance = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img_feats, img_masks=None):
        # 投影到目标维度
        x = self.input_proj(img_feats)
        
        # 多层特征增强
        for layer in self.enhance_layers:
            x = layer(x) + x  # 残差连接
            
        # 计算特征重要性权重
        importance_weights = self.feature_importance(x)
        
        # 应用重要性权重
        enhanced_feats = x * importance_weights

        if img_masks is not None:
            mask = img_masks.unsqueeze(-1)
            enhanced_feats = enhanced_feats * mask
        
        return enhanced_feats


class AdvancedAttributeFeatureExtractor(nn.Module):
    """高级属性特征提取器，目标mAP > 0.8"""
    
    def __init__(
        self,
        feat_dim,
        num_attributes=40,
        d_model=512,
        nhead=8,
        num_layers=6,  # 使用6层Transformer
        dim_feedforward=2048,
        dropout=0.1,
        class_weights=None,
        loss_type='focal',  # 默认使用Focal Loss
        focal_gamma=2.0,
        use_feature_enhancement=True,
        use_relative_pos=True
    ):
        super().__init__()
        self.num_attributes = num_attributes
        self.d_model = d_model
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        
        # 特征增强模块
        if use_feature_enhancement:
            self.feature_enhancer = FeatureEnhancementModule(feat_dim, d_model)
        else:
            self.feature_enhancer = nn.Linear(feat_dim, d_model)
        
        # 属性嵌入（添加可学习的偏置）
        self.attr_embeddings = nn.Parameter(torch.randn(num_attributes, d_model))
        self.attr_bias = nn.Parameter(torch.zeros(num_attributes, 1))
        
        # 位置编码
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_len=100)
        
        # 6层增强Transformer
        self.layers = nn.ModuleList([
            EnhancedAttributeTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_relative_pos=use_relative_pos
            ) for _ in range(num_layers)
        ])
        
        # 增强的MLP头部
        self.attribute_interaction = nn.MultiheadAttention(
            d_model, nhead//2, dropout=dropout, batch_first=True
        )
        self.attr_interact_norm = nn.LayerNorm(d_model)
        self.attr_interact_dropout = nn.Dropout(dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout // 2),
            nn.Linear(dim_feedforward // 2, 1),
        )
        
        # 输出校准层
        self.output_calibration = nn.Sequential(
            nn.Linear(num_attributes, num_attributes // 2),
            nn.ReLU(),
            nn.Linear(num_attributes // 2, num_attributes),
            nn.Sigmoid()
        )
        
        # 损失函数配置
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
            
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        # 属性嵌入特殊初始化
        nn.init.normal_(self.attr_embeddings, mean=0.0, std=0.02)

    def forward_logits(self, img_feats, img_masks=None):
        # 特征增强
        if isinstance(self.feature_enhancer, FeatureEnhancementModule):
            x = self.feature_enhancer(img_feats, img_masks)
        else:
            x = self.feature_enhancer(img_feats)
            
        if img_masks is not None:
            key_padding_mask = img_masks == 0
        else:
            key_padding_mask = None
            
        batch_size = x.size(0)
        
        # 属性token准备
        attr_tokens = self.attr_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 添加位置编码
        pos_enc = self.pos_encoding(attr_tokens)
        
        # 6层Transformer处理
        attention_weights = []
        for i, layer in enumerate(self.layers):
            attr_tokens, attn_weights = layer(
                attr_tokens, x, key_padding_mask=key_padding_mask, pos_encoding=pos_enc
            )
            attention_weights.append(attn_weights)
        
        # 属性间交互注意力
        interacted_attrs, _ = self.attribute_interaction(
            attr_tokens, attr_tokens, attr_tokens
        )
        attr_tokens = attr_tokens + self.attr_interact_dropout(interacted_attrs)
        attr_tokens = self.attr_interact_norm(attr_tokens)
        
        # MLP处理每个属性
        logits_list = []
        for i in range(self.num_attributes):
            attr_feat = attr_tokens[:, i, :]  # 取单个属性的特征
            logit = self.mlp(attr_feat)
            logits_list.append(logit)
        
        logits = torch.cat(logits_list, dim=1)
        
        # 添加属性偏置
        logits = logits + self.attr_bias.squeeze().unsqueeze(0)
        
        return logits, attention_weights

    def forward(self, img_feats, img_masks=None):
        logits, _ = self.forward_logits(img_feats, img_masks)
        probs = torch.sigmoid(logits)
        
        # 输出校准
        calibrated_probs = self.output_calibration(probs)
        
        return calibrated_probs

    def compute_loss(self, img_feats, targets, img_masks=None, class_weights=None, reduction="mean"):
        logits, attention_weights = self.forward_logits(img_feats, img_masks)
        
        # 增强的损失计算
        if self.loss_type == 'focal':
            alpha = class_weights if class_weights is not None else self.class_weights
            criterion = EnhancedFocalLoss(
                alpha=alpha, 
                gamma=self.focal_gamma,
                reduction=reduction,
                adaptive_gamma=True
            )
        else:
            if class_weights is not None:
                cw = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=cw, reduction=reduction)
            elif self.class_weights is not None:
                criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights, reduction=reduction)
            else:
                criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        
        main_loss = criterion(logits, targets)
        
        return main_loss


class LearnedPositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return self.position_embedding(positions)


# 训练配置建议
def get_optimizer_config(model):
    """返回优化的训练配置"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    return optimizer, scheduler


def create_model(feat_dim, num_attributes=40, class_weights=None):
    """创建优化后的模型实例"""
    return AdvancedAttributeFeatureExtractor(
        feat_dim=feat_dim,
        num_attributes=num_attributes,
        d_model=512,
        nhead=8,
        num_layers=6,  # 6层Transformer
        dim_feedforward=2048,
        dropout=0.1,
        class_weights=class_weights,
        loss_type='focal',
        focal_gamma=2.0,
        use_feature_enhancement=True,
        use_relative_pos=True
    )


class AttributeFeatureExtractor(AdvancedAttributeFeatureExtractor):
    """向后兼容，旧代码可继续通过 AttributeFeatureExtractor 导入。"""

    def __init__(
        self,
        feat_dim,
        num_attributes=40,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        class_weights=None,
        loss_type='focal',
        focal_gamma=2.0,
        use_feature_enhancement=True,
        use_relative_pos=True,
    ):
        super().__init__(
            feat_dim=feat_dim,
            num_attributes=num_attributes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            class_weights=class_weights,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            use_feature_enhancement=use_feature_enhancement,
            use_relative_pos=use_relative_pos,
        )
