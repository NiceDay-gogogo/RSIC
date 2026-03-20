#!/usr/bin/env python
"""
测试 MultiAttrAttentionModel 模型是否能正确初始化和前向传播
"""

import torch
import sys
import argparse

# 添加项目路径
sys.path.insert(0, '.')

from models.MultiAttrAttentionModel import MultiAttrAttentionModel, MultiLevelAttributeAttention

def test_multi_level_attention():
    """测试多层次属性注意力模块"""
    print("\n" + "="*50)
    print("测试 MultiLevelAttributeAttention 模块")
    print("="*50)
    
    # 创建模块
    module = MultiLevelAttributeAttention(
        visual_dim=512,
        attr_dim=25,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1
    )
    
    # 创建测试数据
    batch_size = 4
    visual_feats = torch.randn(batch_size, 512)
    attr_feats = torch.randn(batch_size, 25)
    
    # 前向传播
    print(f"\n输入维度:")
    print(f"  visual_feats: {visual_feats.shape}")
    print(f"  attr_feats: {attr_feats.shape}")
    
    fused_features = module(visual_feats, attr_feats)
    
    print(f"\n输出维度:")
    print(f"  fused_features: {fused_features.shape}")
    print(f"  预期: torch.Size([{batch_size}, 512])")
    
    assert fused_features.shape == (batch_size, 512), "输出维度不正确!"
    print("\n✅ 多层次属性注意力模块测试通过!")
    
    return module


def test_full_model():
    """测试完整模型"""
    print("\n" + "="*50)
    print("测试 MultiAttrAttentionModel 完整模型")
    print("="*50)
    
    # 创建模拟的 opt 对象
    class MockOpt:
        def __init__(self):
            self.vocab_size = 1000
            self.input_encoding_size = 512
            self.rnn_type = 'lstm'
            self.rnn_size = 512
            self.num_layers = 2
            self.drop_prob_lm = 0.5
            self.seq_length = 16
            self.fc_feat_size = 2048
            self.attr_feat_size = 25
            self.attr_lambda = 1.0
            self.attr_hidden_dim = 512
            self.attr_num_heads = 8
            self.vocab = {str(i): f'word_{i}' for i in range(1000)}
    
    opt = MockOpt()
    
    # 创建模型
    print("\n正在创建模型...")
    model = MultiAttrAttentionModel(opt)
    model.eval()  # 设置为评估模式
    
    print(f"✓ 模型创建成功")
    print(f"  参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建测试数据
    batch_size = 4
    seq_length = 16
    
    fc_feats = torch.randn(batch_size, 2048)
    att_feats = torch.randn(batch_size, 196, 2048)  # 不使用，但需要传入
    attr_feats = torch.randn(batch_size, 25)
    seq = torch.randint(0, 1000, (batch_size, seq_length + 1))
    
    print(f"\n输入维度:")
    print(f"  fc_feats: {fc_feats.shape}")
    print(f"  attr_feats: {attr_feats.shape}")
    print(f"  seq: {seq.shape}")
    
    # 测试训练模式的前向传播
    print(f"\n测试训练模式前向传播...")
    with torch.no_grad():
        output = model._forward(fc_feats, att_feats, seq, attr_feats=attr_feats)
    
    print(f"  输出维度: {output.shape}")
    print(f"  预期: torch.Size([{batch_size}, {seq_length}, {opt.vocab_size + 1}])")
    
    expected_shape = (batch_size, seq_length, opt.vocab_size + 1)
    assert output.shape == expected_shape, f"输出维度不正确! 期望 {expected_shape}, 得到 {output.shape}"
    print(f"  ✓ 训练模式前向传播成功")
    
    # 测试采样模式
    print(f"\n测试采样模式...")
    with torch.no_grad():
        seq_out, seqLogprobs = model._sample(
            fc_feats, att_feats, attr_feats=attr_feats, 
            opt={'sample_method': 'greedy'}
        )
    
    print(f"  seq_out: {seq_out.shape}")
    print(f"  seqLogprobs: {seqLogprobs.shape}")
    print(f"  ✓ 采样模式测试成功")
    
    print("\n✅ 完整模型测试通过!")
    
    return model


def print_model_architecture(model):
    """打印模型架构"""
    print("\n" + "="*50)
    print("模型架构详情")
    print("="*50)
    
    print("\n【多层次属性注意力模块】")
    print(model.multi_level_attr_attention)
    
    print("\n【ShowTell 基础组件】")
    print(f"  fc_to_visual: {model.fc_to_visual}")
    print(f"  img_embed: {model.img_embed}")
    print(f"  core (LSTM): {model.core}")
    print(f"  embed: {model.embed}")
    print(f"  logit: {model.logit}")


if __name__ == '__main__':
    print("\n" + "🚀"*25)
    print("MultiAttrAttentionModel 单元测试")
    print("🚀"*25)
    
    try:
        # 测试多层次注意力模块
        attention_module = test_multi_level_attention()
        
        # 测试完整模型
        full_model = test_full_model()
        
        # 打印架构
        print_model_architecture(full_model)
        
        print("\n" + "✅"*25)
        print("所有测试通过! 模型可以正常使用!")
        print("✅"*25 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
