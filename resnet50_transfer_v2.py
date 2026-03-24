# finetune.py
"""
猫狗分类 - 微调卷积层
在已有训练好的模型基础上，解冻部分卷积层进行微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 复用已有的类
class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt',optimizer=None, epoch=0):
        """
        Args:
            patience: 验证集性能不再提升的等待轮数
            verbose: 是否打印早停信息
            delta: 认为有提升的最小变化
            path: 模型保存路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf 
        self.delta = delta
        self.path = path
        self.optimizer = optimizer
        self.epoch = epoch
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """保存完整检查点"""
        if self.verbose:
            print(f'验证损失下降 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
        
        # 保存完整检查点，与save_model一致
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'val_loss': val_loss,
            'epoch': self.epoch,
            'best_score': self.best_score
        }
        
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss

class LRSchedulerCallback:
    """学习率调度回调"""
    def __init__(self, optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        
        self.best = None
        self.counter = 0
        self.lr_history = []
        
    def step(self, metrics):
        """更新学习率"""
        if self.best is None:
            self.best = metrics
            return
        
        if (self.mode == 'min' and metrics < self.best) or \
           (self.mode == 'max' and metrics > self.best):
            self.best = metrics
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.reduce_lr()
                self.counter = 0
    
    def reduce_lr(self):
        """降低学习率"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            self.lr_history.append(new_lr)
            
            if self.verbose:
                print(f'学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}')

class CatDogDataset(Dataset):
    """猫狗数据集类"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label, class_name in enumerate(['cat', 'dog']):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"在 {data_dir} 中没有找到图片文件")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"警告: 无法加载图片 {img_path}, 错误: {e}")
            dummy_image = torch.randn(3, 224, 224)
            return dummy_image, torch.tensor(0, dtype=torch.float32)

class CatDogFinetuner:
    """猫狗分类微调器"""
    def __init__(self, num_classes=2, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.early_stopping = None
        self.lr_scheduler = None
    
    def create_data_transforms(self):
        """创建数据转换（与原始训练相同）"""
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def load_pretrained_model(self, model_path, unfreeze_layers=5):
        print(f"加载已训练模型: {model_path}")
        
        try:
            # 创建模型
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
            
            # 尝试加载
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 直接尝试加载，不判断格式
            try:
                # 先尝试作为完整字典
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ 加载为完整字典格式")
            except:
                try:
                    # 再尝试作为状态字典
                    model.load_state_dict(checkpoint)
                    print("✅ 加载为状态字典格式")
                except:
                    # 最后尝试，可能是保存的完整模型
                    if hasattr(checkpoint, 'state_dict'):
                        model.load_state_dict(checkpoint.state_dict())
                        print("✅ 加载为模型对象格式")
                    else:
                        raise ValueError("无法识别的模型格式")
            
            model = model.to(self.device)
            print("✅ 模型加载成功")
            
            # 解冻指定层数
            self._unfreeze_layers(model, unfreeze_layers)
            
            self.model = model
            return self.model
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def _unfreeze_layers(self, model, unfreeze_layers=5):
        print(f"解冻最后 {unfreeze_layers} 层进行微调...")
        
        # 先冻结所有层
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻全连接层（必须解冻，因为要继续训练）
        for param in model.fc.parameters():
            param.requires_grad = True
        
        # 收集所有可解冻的层
        all_layers = []
        
        # 从后往前收集
        # layer4 (最后的卷积块) 优先
        for name, param in model.layer4.named_parameters():
            all_layers.append((f"layer4.{name}", param))
        
        # layer3
        for name, param in model.layer3.named_parameters():
            all_layers.append((f"layer3.{name}", param))
        
        # layer2
        for name, param in model.layer2.named_parameters():
            all_layers.append((f"layer2.{name}", param))
        
        # 只解冻指定数量的层
        layers_to_unfreeze = all_layers[:unfreeze_layers]
        
        # 解冻选中的层
        for name, param in layers_to_unfreeze:
            param.requires_grad = True
            print(f"  ✅ 解冻: {name}")
        
        # 统计参数
        self._print_trainable_params(model)
        
        return len(layers_to_unfreeze)
    
    def _print_trainable_params(self, model):
        """打印可训练参数信息"""
        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                # print(f"  ✓ 可训练: {name}")
        
        print(f"📊 参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params:.2%})")
        print(f"  冻结参数: {total_params - trainable_params:,} ({(total_params - trainable_params)/total_params:.2%})")
    
    def create_data_loaders(self, data_dir, batch_size=32):
        """创建数据加载器"""
        train_transform, val_transform = self.create_data_transforms()
        
        try:
            train_dataset = CatDogDataset(
                os.path.join(data_dir, 'train'),
                transform=train_transform
            )
            
            val_dataset = CatDogDataset(
                os.path.join(data_dir, 'val'),
                transform=val_transform
            )
            
            test_dataset = CatDogDataset(
                os.path.join(data_dir, 'test'),
                transform=val_transform
            )
            
            if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
                raise ValueError("数据集为空，请检查数据路径和文件")
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            print(f"❌ 创建数据加载器失败: {e}")
            raise
    
    def finetune(self, train_loader, val_loader, epochs=10, learning_rate=0.0001, 
                 patience_early_stop=4, patience_lr=3, model_save_path='finetuned_model.pth'):
        """
        微调模型
        
        Args:
            epochs: 微调轮次
            learning_rate: 微调学习率（通常比训练时小）
        """
        # 定义损失函数和优化器
        self.criterion = nn.BCELoss()
        
        # 只优化可训练的参数
        params_to_optimize = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(params_to_optimize, lr=learning_rate)
        
        # 初始化早停和学习率回调
        self.early_stopping = EarlyStopping(
            patience=patience_early_stop, 
            verbose=True, 
            path=model_save_path,
            optimizer=self.optimizer,  # 传递优化器
            epoch=0  # 初始epoch
        )
        self.lr_scheduler = LRSchedulerCallback(
            self.optimizer, 
            mode='min',
            patience=patience_lr,
            verbose=True
        )
        
        history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [],
            'train_precision': [], 'train_recall': [], 'train_f1': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [],
            'learning_rate': []
        }
        
        print(f"开始微调，共{epochs}个epoch，学习率: {learning_rate}")
        print(f"早停耐心值: {patience_early_stop}，学习率回调耐心值: {patience_lr}")
        print("="*60)
        
        for epoch in range(epochs):
            self.early_stopping.epoch = epoch + 1
            # 训练阶段
            try:
                train_metrics = self._train_epoch(train_loader, epoch, epochs)
                for key in ['loss', 'acc', 'precision', 'recall', 'f1']:
                    history[f'train_{key}'].append(train_metrics[key])
            except Exception as e:
                print(f"❌ 训练阶段出错: {e}")
                break
            
            # 验证阶段
            try:
                val_metrics, _, _ = self._validate(val_loader)
                for key in ['loss', 'acc', 'precision', 'recall', 'f1']:
                    history[f'val_{key}'].append(val_metrics[key])
            except Exception as e:
                print(f"❌ 验证阶段出错: {e}")
                break
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            # 打印epoch结果
            self._print_epoch_summary(epoch, epochs, train_metrics, val_metrics, current_lr)
            
            # 学习率回调
            self.lr_scheduler.step(val_metrics['loss'])
            
            # 早停检查
            self.early_stopping(val_metrics['loss'], self.model)
            if self.early_stopping.early_stop:
                print(f"⚡ 早停在 epoch {epoch+1}")
                break
        
        # 加载最佳模型
        if os.path.exists(model_save_path):
            self.model.load_state_dict(torch.load(model_save_path, map_location=self.device))
            print(f"✅ 加载最佳模型: {model_save_path}")
        else:
            print("⚠️  没有找到保存的最佳模型")
        
        return history
    
    def _train_epoch(self, train_loader, epoch, total_epochs):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_outputs = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [微调]')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            try:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                all_outputs.append(outputs.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
            except Exception as e:
                print(f"❌ 批次 {batch_idx} 训练出错: {e}")
                continue
        
        if len(all_outputs) > 0:
            all_outputs = np.concatenate(all_outputs)
            all_labels = np.concatenate(all_labels)
            metrics = self._compute_metrics(all_outputs, all_labels, total_loss / len(train_loader.dataset))
        else:
            metrics = {'loss': 0, 'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        return metrics
    
    def _validate(self, val_loader):
        """验证"""
        self.model.eval()
        val_loss = 0.0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="验证")
            for inputs, labels in pbar:
                try:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).unsqueeze(1)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    all_outputs.append(outputs.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"❌ 验证批次出错: {e}")
                    continue
        
        if len(all_outputs) > 0:
            all_outputs = np.concatenate(all_outputs)
            all_labels = np.concatenate(all_labels)
            metrics = self._compute_metrics(all_outputs, all_labels, val_loss / len(val_loader.dataset))
        else:
            metrics = {'loss': 0, 'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            all_outputs = np.array([])
            all_labels = np.array([])
        
        return metrics, all_outputs, all_labels
    
    def _compute_metrics(self, outputs, labels, loss):
        """计算所有指标"""
        if len(outputs) == 0 or len(labels) == 0:
            return {'loss': loss, 'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        predictions = (outputs > 0.5).astype(int)
        labels_flat = labels.flatten()
        predictions_flat = predictions.flatten()
        
        try:
            accuracy = accuracy_score(labels_flat, predictions_flat)
            precision = precision_score(labels_flat, predictions_flat, zero_division=0)
            recall = recall_score(labels_flat, predictions_flat, zero_division=0)
            f1 = f1_score(labels_flat, predictions_flat, zero_division=0)
        except Exception as e:
            print(f"❌ 计算指标出错: {e}")
            accuracy = precision = recall = f1 = 0
        
        metrics = {
            'loss': loss,
            'acc': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def _print_epoch_summary(self, epoch, total_epochs, train_metrics, val_metrics, lr):
        """打印epoch总结"""
        print(f"\n📊 Epoch {epoch+1}/{total_epochs} 总结:")
        print("-" * 50)
        print(f"学习率: {lr:.6f}")
        print(f"训练 - 损失: {train_metrics['loss']:.4f}, "
              f"准确率: {train_metrics['acc']:.4f}, "
              f"精确率: {train_metrics['precision']:.4f}, "
              f"召回率: {train_metrics['recall']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"验证 - 损失: {val_metrics['loss']:.4f}, "
              f"准确率: {val_metrics['acc']:.4f}, "
              f"精确率: {val_metrics['precision']:.4f}, "
              f"召回率: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        print("-" * 50)
    
    def evaluate(self, test_loader, return_predictions=False):
        """评估模型（复用原有评估方法）"""
        self.model.eval()
        all_outputs = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="测试")
            for inputs, labels in pbar:
                try:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).unsqueeze(1)
                    
                    outputs = self.model(inputs)
                    predictions = (outputs > 0.5).float()
                    
                    all_outputs.append(outputs.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    all_predictions.append(predictions.cpu().numpy())
                    
                except Exception as e:
                    print(f"❌ 测试批次出错: {e}")
                    continue
        
        if len(all_outputs) == 0:
            print("❌ 没有可用的测试数据")
            if return_predictions:
                return {}, [], [], []
            return {}
        
        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)
        
        y_true = all_labels.flatten()
        y_pred = all_predictions.flatten()
        y_prob = all_outputs.flatten()
        
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            print(f"❌ 计算指标出错: {e}")
            accuracy = precision = recall = f1 = 0
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.0
        
        try:
            loss = float(self.criterion(
                torch.tensor(y_prob).unsqueeze(1), 
                torch.tensor(y_true).unsqueeze(1)
            ).item())
        except:
            loss = 0.0
        
        try:
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
        except:
            cm = np.array([[0, 0], [0, 0]])
            tn = fp = fn = tp = 0
        
        try:
            class_report = classification_report(y_true, y_pred, 
                                               target_names=['猫', '狗'], 
                                               output_dict=True)
        except:
            class_report = {
                '猫': {'precision': 0, 'recall': 0, 'f1-score': 0},
                '狗': {'precision': 0, 'recall': 0, 'f1-score': 0},
                'weighted avg': {'f1-score': 0}
            }
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp
        }
        
        self._print_evaluation_summary(metrics)
        
        if return_predictions:
            return metrics, y_true, y_pred, y_prob
        else:
            return metrics
    
    def _print_evaluation_summary(self, metrics):
        """打印评估总结"""
        print("\n" + "="*60)
        print("模型评估结果")
        print("="*60)
        
        print(f"\n📊 性能指标:")
        print(f"   损失: {metrics['loss']:.4f}")
        print(f"   准确率: {metrics['accuracy']:.4f}")
        print(f"   精确率: {metrics['precision']:.4f}")
        print(f"   召回率: {metrics['recall']:.4f}")
        print(f"   F1分数: {metrics['f1_score']:.4f}")
        print(f"   AUC: {metrics['auc']:.4f}")
        
        print(f"\n📈 混淆矩阵:")
        print(f"            预测猫    预测狗")
        print(f"   真实猫    {metrics['true_negative']:6d}    {metrics['false_positive']:6d}")
        print(f"   真实狗    {metrics['false_negative']:6d}    {metrics['true_positive']:6d}")
        
        print(f"\n🔍 分类报告:")
        report = metrics['classification_report']
        if '猫' in report and '狗' in report:
            print(f"   猫 - 精确率: {report['猫']['precision']:.4f}, "
                  f"召回率: {report['猫']['recall']:.4f}, "
                  f"F1: {report['猫']['f1-score']:.4f}")
            print(f"   狗 - 精确率: {report['狗']['precision']:.4f}, "
                  f"召回率: {report['狗']['recall']:.4f}, "
                  f"F1: {report['狗']['f1-score']:.4f}")
            print(f"   加权平均F1: {report['weighted avg']['f1-score']:.4f}")
    
    def plot_training_history(self, history, figsize=(15, 10)):
        """绘制训练历史"""
        if not history or 'train_loss' not in history or len(history['train_loss']) == 0:
            print("❌ 没有训练历史数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. 损失曲线
        ax1 = axes[0, 0]
        ax1.plot(history['train_loss'], 'b-', label='训练损失', linewidth=2, alpha=0.7)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            ax1.plot(history['val_loss'], 'r-', label='验证损失', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        ax2 = axes[0, 1]
        ax2.plot(history['train_acc'], 'b-', label='训练准确率', linewidth=2, alpha=0.7)
        if 'val_acc' in history and len(history['val_acc']) > 0:
            ax2.plot(history['val_acc'], 'r-', label='验证准确率', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率')
        ax2.set_title('准确率曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 学习率变化
        ax3 = axes[0, 2]
        if 'learning_rate' in history and len(history['learning_rate']) > 0:
            ax3.plot(history['learning_rate'], 'g-', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('学习率')
            ax3.set_title('学习率变化')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '无学习率数据', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # 4. 精确率曲线
        ax4 = axes[1, 0]
        if 'train_precision' in history and len(history['train_precision']) > 0:
            ax4.plot(history['train_precision'], 'b-', label='训练精确率', linewidth=2, alpha=0.7)
        if 'val_precision' in history and len(history['val_precision']) > 0:
            ax4.plot(history['val_precision'], 'r-', label='验证精确率', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('精确率')
        ax4.set_title('精确率曲线')
        if 'train_precision' in history or 'val_precision' in history:
            ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 召回率曲线
        ax5 = axes[1, 1]
        if 'train_recall' in history and len(history['train_recall']) > 0:
            ax5.plot(history['train_recall'], 'b-', label='训练召回率', linewidth=2, alpha=0.7)
        if 'val_recall' in history and len(history['val_recall']) > 0:
            ax5.plot(history['val_recall'], 'r-', label='验证召回率', linewidth=2, alpha=0.7)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('召回率')
        ax5.set_title('召回率曲线')
        if 'train_recall' in history or 'val_recall' in history:
            ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. F1分数曲线
        ax6 = axes[1, 2]
        if 'train_f1' in history and len(history['train_f1']) > 0:
            ax6.plot(history['train_f1'], 'b-', label='训练F1', linewidth=2, alpha=0.7)
        if 'val_f1' in history and len(history['val_f1']) > 0:
            ax6.plot(history['val_f1'], 'r-', label='验证F1', linewidth=2, alpha=0.7)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('F1分数')
        ax6.set_title('F1分数曲线')
        if 'train_f1' in history or 'val_f1' in history:
            ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('微调训练历史')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm, figsize=(8, 6)):
        """绘制混淆矩阵"""
        if cm is None or cm.size == 0 or cm.sum() == 0:
            print("❌ 混淆矩阵为空，无法绘制")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['猫', '狗'], yticklabels=['猫', '狗'], ax=ax)
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        ax.set_title('混淆矩阵')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path):
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            }, path)
            print(f"✅ 模型已保存到: {path}")
        except Exception as e:
            print(f"❌ 保存模型失败: {e}")

def main_finetune():
    """微调主函数"""
    data_dir = "E:/AI_Projects/Cat_Dog_Classify/split_data"
    pretrained_model = "best_cat_dog_model.pth"  # 已训练好的模型
    
    print("="*60)
    print("猫狗分类器 - 微调卷积层")
    print("="*60)
    
    # 检查预训练模型是否存在
    if not os.path.exists(pretrained_model):
        print(f"❌ 找不到预训练模型: {pretrained_model}")
        print("请先运行 train_basic.py 训练基础模型")
        return
    
    # 创建微调器
    print("初始化微调器...")
    finetuner = CatDogFinetuner()
    
    # 加载已训练模型并解冻层
    print(f"加载模型并解冻层...")
    try:
        model = finetuner.load_pretrained_model(
            model_path=pretrained_model,
            unfreeze_layers=5  # 解冻最后n层
        )
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 创建数据加载器
    print("加载数据集...")
    try:
        train_loader, val_loader, test_loader = finetuner.create_data_loaders(
            data_dir=data_dir,
            batch_size=32
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    print(f"\n📊 数据集统计:")
    print(f"  训练集: {len(train_loader.dataset)} 张图片")
    print(f"  验证集: {len(val_loader.dataset)} 张图片")
    print(f"  测试集: {len(test_loader.dataset)} 张图片")
    
    # 微调模型
    print("\n🎯 开始微调...")
    try:
        history = finetuner.finetune(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5,              # 微调轮次较少
            learning_rate=0.0001,  # 更小的学习率
            patience_early_stop=3,
            patience_lr=3,
            model_save_path='finetuned_cat_dog_model.pth'
        )
    except Exception as e:
        print(f"❌ 微调失败: {e}")
        return
    
    # 绘制训练历史
    print("\n📈 绘制微调历史...")
    finetuner.plot_training_history(history)
    
    # 在测试集上评估
    print("\n🧪 在测试集上评估...")
    try:
        test_metrics, y_true, y_pred, y_prob = finetuner.evaluate(
            test_loader, 
            return_predictions=True
        )
    except Exception as e:
        print(f"❌ 测试评估失败: {e}")
        test_metrics = {}
    
    # 绘制混淆矩阵
    print("\n📊 绘制混淆矩阵...")
    if 'confusion_matrix' in test_metrics:
        finetuner.plot_confusion_matrix(test_metrics['confusion_matrix'])
    
    # 保存最终模型
    finetuner.save_model("final_finetuned_model.pth")
    
    # 生成微调报告
    generate_finetune_report(test_metrics, history, unfreeze_layers=5)
    
    print("\n🎉 微调完成！")

def generate_finetune_report(test_metrics, history, unfreeze_layers=5, 
                           report_path='finetune_report.txt'):
    """生成微调报告"""
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("猫狗分类模型 - 微调评估报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("🎯 微调配置:\n")
            f.write("="*40 + "\n")
            f.write(f"   解冻层数: {unfreeze_layers}\n")
            f.write(f"   微调轮次: {len(history.get('train_loss', []))}\n")
            f.write(f"   最终学习率: {history.get('learning_rate', [0])[-1]:.6f}\n\n")
            
            f.write("📊 微调后性能指标:\n")
            f.write("="*40 + "\n")
            f.write(f"   准确率: {test_metrics.get('accuracy', 0):.4f}\n")
            f.write(f"   精确率: {test_metrics.get('precision', 0):.4f}\n")
            f.write(f"   召回率: {test_metrics.get('recall', 0):.4f}\n")
            f.write(f"   F1分数: {test_metrics.get('f1_score', 0):.4f}\n")
            f.write(f"   AUC: {test_metrics.get('auc', 0):.4f}\n\n")
            
            f.write("📈 训练过程分析:\n")
            f.write("="*40 + "\n")
            
            if 'train_loss' in history and 'val_loss' in history:
                initial_train_loss = history['train_loss'][0]
                final_train_loss = history['train_loss'][-1]
                initial_val_loss = history['val_loss'][0]
                final_val_loss = history['val_loss'][-1]
                
                f.write(f"   初始训练损失: {initial_train_loss:.4f}\n")
                f.write(f"   最终训练损失: {final_train_loss:.4f}\n")
                f.write(f"   训练损失下降: {initial_train_loss - final_train_loss:.4f}\n")
                f.write(f"   初始验证损失: {initial_val_loss:.4f}\n")
                f.write(f"   最终验证损失: {final_val_loss:.4f}\n")
                f.write(f"   验证损失下降: {initial_val_loss - final_val_loss:.4f}\n\n")
                
                # 分析过拟合
                train_val_gap = abs(final_train_loss - final_val_loss)
                if train_val_gap > 0.1:
                    f.write(f"   ⚠️  存在过拟合 (训练-验证差距: {train_val_gap:.4f})\n")
                else:
                    f.write(f"   ✅ 拟合良好 (训练-验证差距: {train_val_gap:.4f})\n")
            
            f.write("\n💡 微调效果分析:\n")
            f.write("="*40 + "\n")
            
            accuracy = test_metrics.get('accuracy', 0)
            if accuracy >= 0.9:
                f.write("   微调效果显著，模型性能优秀\n")
                f.write("   建议: 可考虑将模型部署到生产环境\n")
            elif accuracy >= 0.85:
                f.write("   微调有一定效果，模型性能良好\n")
                f.write("   建议: 可尝试解冻更多层进行进一步微调\n")
            else:
                f.write("   微调效果有限，可尝试不同策略\n")
                f.write("   建议:\n")
                f.write("     1. 调整解冻层数\n")
                f.write("     2. 调整学习率\n")
                f.write("     3. 增加微调轮次\n")
        
        print(f"✅ 微调报告已保存到: {report_path}")
        
    except Exception as e:
        print(f"❌ 生成报告失败: {e}")

if __name__ == "__main__":
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 运行微调
    main_finetune()