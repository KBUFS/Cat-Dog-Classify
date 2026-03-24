"""
数据集质量检查模块
"""
import os
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def check_dataset_quality(data_dir):
    """
    全面检查数据集质量
    """
    print("="*60)
    print("数据集质量检查")
    print("="*60)
    
    classes = ['cat', 'dog']
    splits = ['train', 'val', 'test']
    
    stats = {}
    
    for split in splits:
        print(f"\n--- {split.upper()} 集统计 ---")
        split_stats = {}
        
        for cls in classes:
            cls_dir = os.path.join(data_dir, split, cls)
            if not os.path.exists(cls_dir):
                print(f"警告: {cls_dir} 不存在")
                continue
            
            images = [f for f in os.listdir(cls_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            
            if len(images) == 0:
                print(f"警告: {cls_dir} 没有图片")
                continue
            
            # 检查每个图片
            valid_images = 0
            corrupted_images = []
            sizes = []
            aspect_ratios = []
            
            for img_file in images:
                img_path = os.path.join(cls_dir, img_file)
                try:
                    # 尝试打开图片
                    with Image.open(img_path) as img:
                        img.verify()  # 验证图片完整性
                        img = Image.open(img_path)  # 重新打开
                        
                        # 获取图片信息
                        width, height = img.size
                        sizes.append((width, height))
                        aspect_ratios.append(width / height if height > 0 else 0)
                        valid_images += 1
                        
                except Exception as e:
                    corrupted_images.append(img_file)
                    print(f"损坏图片: {img_path} - 错误: {str(e)}")
            
            split_stats[cls] = {
                'count': len(images),
                'valid': valid_images,
                'corrupted': corrupted_images,
                'sizes': sizes,
                'aspect_ratios': aspect_ratios
            }
            
            print(f"  {cls}: {valid_images} 张有效图片")
            if corrupted_images:
                print(f"  {cls}: {len(corrupted_images)} 张损坏图片")
        
        stats[split] = split_stats
    
    # 可视化统计
    visualize_dataset_stats(stats)
    
    return stats

def visualize_dataset_stats(stats):
    """
    可视化数据集统计
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 各类别数量
    for idx, split in enumerate(['train', 'val', 'test']):
        if split in stats:
            ax = axes[0, idx]
            split_stats = stats[split]
            
            classes = list(split_stats.keys())
            counts = [split_stats[cls]['count'] for cls in classes]
            
            bars = ax.bar(classes, counts)
            ax.set_title(f'{split} 集 - 类别分布')
            ax.set_ylabel('图片数量')
            
            # 在柱子上显示数字
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom')
    
    # 2. 图片尺寸分布
    for idx, split in enumerate(['train', 'val', 'test']):
        if split in stats:
            ax = axes[1, idx]
            split_stats = stats[split]
            
            all_widths = []
            all_heights = []
            
            for cls in split_stats:
                sizes = split_stats[cls]['sizes']
                if sizes:
                    widths, heights = zip(*sizes)
                    all_widths.extend(widths)
                    all_heights.extend(heights)
            
            if all_widths and all_heights:
                ax.scatter(all_widths, all_heights, alpha=0.5, s=10)
                ax.set_xlabel('宽度')
                ax.set_ylabel('高度')
                ax.set_title(f'{split} 集 - 图片尺寸分布')
                
                # 添加平均尺寸线
                avg_width = np.mean(all_widths)
                avg_height = np.mean(all_heights)
                ax.axvline(avg_width, color='r', linestyle='--', alpha=0.5)
                ax.axhline(avg_height, color='r', linestyle='--', alpha=0.5)
                ax.text(0.05, 0.95, f'平均: {int(avg_width)}×{int(avg_height)}', 
                       transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    # 3. 计算并显示不均衡程度
    print("\n" + "="*60)
    print("类别平衡性分析")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        if split in stats:
            split_stats = stats[split]
            
            if len(split_stats) >= 2:  # 至少有2个类别
                cat_count = split_stats.get('cat', {}).get('valid', 0)
                dog_count = split_stats.get('dog', {}).get('valid', 0)
                
                if cat_count > 0 and dog_count > 0:
                    ratio = max(cat_count, dog_count) / min(cat_count, dog_count)
                    
                    print(f"\n{split} 集:")
                    print(f"  猫: {cat_count} 张")
                    print(f"  狗: {dog_count} 张")
                    print(f"  类别比: {ratio:.2f}:1")
                    
                    if ratio > 3:
                        print(f"  ⚠️ 警告: 类别严重不均衡 (比率 > 3:1)")
                    elif ratio > 2:
                        print(f"  ⚠️ 注意: 类别存在不均衡 (比率 > 2:1)")
                    else:
                        print(f"  ✅ 类别相对均衡")

if __name__ == "__main__":
    data_dir = "E:/AI_Projects/Cat_Dog_Classify/split_data"
    stats = check_dataset_quality(data_dir)