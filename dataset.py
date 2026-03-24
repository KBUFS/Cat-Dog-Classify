import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_dataset_fixed(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    修复版数据集拆分
    """
    random.seed(seed)
    
    # 自动计算测试集比例
    test_ratio = 1.0 - train_ratio - val_ratio
    
    classes = ['cat', 'dog']
    
    # 创建目录
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
    
    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        
        if not os.path.exists(cls_dir):
            continue
            
        all_images = [f for f in os.listdir(cls_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        
        print(f"{cls}: 共找到 {len(all_images)} 张图片")
        
        if len(all_images) == 0:
            continue
        
        # 修复1：先打乱
        random.shuffle(all_images)
        
        # 修复2：直接按比例分割，不用两次train_test_split
        total = len(all_images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # 分割
        train_images = all_images[:train_end]
        val_images = all_images[train_end:val_end]
        test_images = all_images[val_end:]
        
        # 复制文件
        for split_name, img_list in [('train', train_images), 
                                     ('val', val_images), 
                                     ('test', test_images)]:
            for img in img_list:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(output_dir, split_name, cls, img)
                shutil.copy2(src, dst)
        
        print(f"  train: {len(train_images)} 张 ({len(train_images)/total:.1%})")
        print(f"  val: {len(val_images)} 张 ({len(val_images)/total:.1%})")
        print(f"  test: {len(test_images)} 张 ({len(test_images)/total:.1%})")

# 运行
if __name__ == "__main__":
    split_dataset_fixed(
        source_dir="E:/AI_Projects/Cat_Dog_Classify/raw_data",
        output_dir="E:/AI_Projects/Cat_Dog_Classify/split_data",
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )