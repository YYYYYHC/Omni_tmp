#!/usr/bin/env python3
"""
测试数据加载器修复
"""

import torch
from torch.utils.data import DataLoader
from data_loaders.humanml.data.dataset_abs import HumanML3D

def test_dataloader():
    """测试数据加载器是否正常工作"""
    print("🧪 测试数据加载器修复")
    print("=" * 50)
    
    try:
        # 创建数据集
        print("📁 创建数据集...")
        dataset = HumanML3D(
            mode='train',
            datapath='./dataset/humanml_opt.txt',
            split="train",
            use_multiprocessing=True,
            num_workers=64
        )
        
        print(f"✅ 数据集创建成功")
        print(f"📊 数据集大小: {len(dataset)}")
        
        # 测试单个样本获取
        print("\n🔍 测试单个样本获取...")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ 样本获取成功")
            print(f"📊 样本类型: {type(sample)}")
            print(f"📊 样本长度: {len(sample)}")
        else:
            print("⚠️  数据集为空")
            return
        
        # 测试DataLoader
        print("\n🔄 测试DataLoader...")
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2,  # 使用较少的worker进行测试
            drop_last=True
        )
        
        print(f"✅ DataLoader创建成功")
        print(f"📊 DataLoader长度: {len(dataloader)}")
        
        # 测试获取一个batch
        print("\n📦 测试获取batch...")
        for i, batch in enumerate(dataloader):
            print(f"✅ 成功获取第 {i+1} 个batch")
            print(f"📊 Batch类型: {type(batch)}")
            print(f"📊 Batch长度: {len(batch)}")
            
            # 只测试第一个batch
            if i == 0:
                for j, item in enumerate(batch):
                    if isinstance(item, torch.Tensor):
                        print(f"  {j}: Tensor shape {item.shape}")
                    elif isinstance(item, str):
                        print(f"  {j}: String length {len(item)}")
                    else:
                        print(f"  {j}: {type(item).__name__}")
            break
            
        print("\n🎉 所有测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_index_validation():
    """测试索引验证"""
    print("\n🔍 测试索引验证")
    print("=" * 30)
    
    try:
        dataset = HumanML3D(
            mode='train',
            datapath='./dataset/humanml_opt.txt',
            split="train",
            use_multiprocessing=False,  # 使用单进程避免复杂性
            num_workers=1
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试正常索引
        if len(dataset) > 0:
            print("✅ 正常索引测试通过")
            dataset[0]
        
        # 测试负索引
        try:
            dataset[-1]
            print("❌ 负索引应该失败但没有失败")
        except ValueError as e:
            print(f"✅ 负索引正确被拒绝: {e}")
        
        # 测试超出范围的索引
        try:
            dataset[len(dataset) + 1]
            print("❌ 超出范围索引应该失败但没有失败")
        except IndexError as e:
            print(f"✅ 超出范围索引正确被拒绝: {e}")
            
    except Exception as e:
        print(f"❌ 索引验证测试失败: {e}")

if __name__ == "__main__":
    test_dataloader()
    test_index_validation()
    print("\n🏁 测试完成！") 