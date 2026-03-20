"""
测试 ShowTellWithAttrModel 是否能正常工作

这个脚本会：
1. 创建一个简单的配置
2. 初始化 ShowTellWithAttrModel
3. 加载属性提取器
4. 测试前向传播和采样
"""
import torch
import argparse
from models.ShowTellWithAttrModel import ShowTellWithAttrModel
from models.AttributeFeatureExtractor import AttributeFeatureExtractor


def create_test_opt():
    """创建测试用的配置对象"""
    opt = argparse.Namespace()
    
    # 基本配置
    opt.vocab_size = 1000  # 假设词表大小为 1000
    opt.input_encoding_size = 512
    opt.rnn_type = 'lstm'
    opt.rnn_size = 512
    opt.num_layers = 1
    opt.drop_prob_lm = 0.5
    opt.seq_length = 20
    opt.fc_feat_size = 2048
    opt.att_feat_size = 2048
    opt.attr_feat_size = 25
    
    # 词表（简化版）
    opt.vocab = {str(i): f'word{i}' for i in range(opt.vocab_size)}
    
    return opt


def test_model_initialization():
    """测试模型初始化"""
    print("=" * 60)
    print("测试 1: 模型初始化")
    print("=" * 60)
    
    opt = create_test_opt()
    model = ShowTellWithAttrModel(opt)
    
    print(f"✓ 模型创建成功")
    print(f"  - 词表大小: {model.vocab_size}")
    print(f"  - RNN 类型: {model.rnn_type}")
    print(f"  - RNN 大小: {model.rnn_size}")
    print(f"  - 属性特征维度: {model.attr_feat_size}")
    print()
    
    return model, opt


def test_forward_pass(model, opt):
    """测试前向传播"""
    print("=" * 60)
    print("测试 2: 前向传播（训练模式）")
    print("=" * 60)
    
    batch_size = 4
    num_regions = 36
    seq_length = 15
    
    # 创建假数据
    fc_feats = torch.randn(batch_size, opt.fc_feat_size)
    att_feats = torch.randn(batch_size, num_regions, opt.att_feat_size)
    attr_feats = torch.randn(batch_size, opt.attr_feat_size)  # 属性特征
    seq = torch.randint(0, opt.vocab_size, (batch_size, seq_length))
    att_masks = torch.ones(batch_size, num_regions)
    
    # 前向传播
    model.train()
    try:
        output = model._forward(fc_feats, att_feats, seq, att_masks, attr_feats)
        print(f"✓ 前向传播成功")
        print(f"  - 输入序列形状: {seq.shape}")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 预期形状: [batch_size, seq_length-1, vocab_size+1]")
        print(f"  - 实际形状: [{output.shape[0]}, {output.shape[1]}, {output.shape[2]}]")
        print()
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sampling(model, opt):
    """测试采样（推理模式）"""
    print("=" * 60)
    print("测试 3: 贪心采样")
    print("=" * 60)
    
    batch_size = 2
    num_regions = 36
    
    # 创建假数据
    fc_feats = torch.randn(batch_size, opt.fc_feat_size)
    att_feats = torch.randn(batch_size, num_regions, opt.att_feat_size)
    attr_feats = torch.randn(batch_size, opt.attr_feat_size)
    att_masks = torch.ones(batch_size, num_regions)
    
    # 贪心采样
    model.eval()
    try:
        with torch.no_grad():
            seq, seqLogprobs = model._sample(
                fc_feats, att_feats, att_masks, attr_feats,
                opt={'sample_method': 'greedy', 'beam_size': 1}
            )
        print(f"✓ 贪心采样成功")
        print(f"  - 生成序列形状: {seq.shape}")
        print(f"  - log 概率形状: {seqLogprobs.shape}")
        print(f"  - 预期: [batch_size, seq_length]")
        print()
        return True
    except Exception as e:
        print(f"✗ 贪心采样失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_attribute_extractor(model, opt):
    """测试与属性提取器集成"""
    print("=" * 60)
    print("测试 4: 与属性提取器集成")
    print("=" * 60)
    
    # 尝试加载属性提取器
    attr_extractor_path = 'save/attribute_extractor_top25.pth'
    
    try:
        import os
        if not os.path.exists(attr_extractor_path):
            print(f"⚠ 属性提取器文件不存在: {attr_extractor_path}")
            print(f"  跳过此测试")
            print()
            return None
        
        print(f"加载属性提取器: {attr_extractor_path}")
        attr_ckpt = torch.load(attr_extractor_path, map_location='cpu')
        attr_args = attr_ckpt.get('args', {})
        
        attr_extractor = AttributeFeatureExtractor(
            feat_dim=attr_args.get('feat_dim', 2048),
            num_attributes=attr_ckpt.get('num_attributes', 25),
            d_model=attr_args.get('d_model', 512),
            nhead=attr_args.get('nhead', 8),
            num_layers=attr_args.get('num_layers', 6),
            dim_feedforward=attr_args.get('dim_feedforward', 2048),
            dropout=attr_args.get('dropout', 0.1),
        )
        attr_extractor.load_state_dict(attr_ckpt['model_state'], strict=False)
        attr_extractor.eval()
        
        print(f"✓ 属性提取器加载成功")
        print(f"  - 属性数量: {attr_extractor.num_attributes}")
        print()
        
        # 测试提取属性特征
        batch_size = 2
        num_regions = 36
        att_feats = torch.randn(batch_size, num_regions, opt.att_feat_size)
        att_masks = torch.ones(batch_size, num_regions)
        
        with torch.no_grad():
            attr_probs = attr_extractor(att_feats, img_masks=att_masks)
        
        print(f"✓ 属性特征提取成功")
        print(f"  - 输入形状: {att_feats.shape}")
        print(f"  - 输出形状: {attr_probs.shape}")
        print(f"  - 预期: [batch_size, num_attributes] = [{batch_size}, {attr_extractor.num_attributes}]")
        print()
        
        return attr_extractor
        
    except Exception as e:
        print(f"✗ 属性提取器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "ShowTellWithAttrModel 测试套件" + " " * 17 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # 测试 1: 模型初始化
    model, opt = test_model_initialization()
    
    # 测试 2: 前向传播
    forward_ok = test_forward_pass(model, opt)
    
    # 测试 3: 采样
    sample_ok = test_sampling(model, opt)
    
    # 测试 4: 属性提取器集成
    attr_extractor = test_with_attribute_extractor(model, opt)
    
    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"模型初始化:       ✓")
    print(f"前向传播:         {'✓' if forward_ok else '✗'}")
    print(f"贪心采样:         {'✓' if sample_ok else '✗'}")
    print(f"属性提取器集成:   {'✓' if attr_extractor is not None else '⚠ (跳过)'}")
    print()
    
    if forward_ok and sample_ok:
        print("🎉 所有核心测试通过！模型可以正常使用。")
        print()
        print("下一步：")
        print("1. 确保属性提取器文件存在: save/attribute_extractor_top25.pth")
        print("2. 使用以下命令开始训练:")
        print("   python train.py --caption_model show_tell_attr --id show_tell_attr_v1")
        print()
    else:
        print("❌ 部分测试失败，请检查错误信息并修复。")
        print()


if __name__ == '__main__':
    main()
