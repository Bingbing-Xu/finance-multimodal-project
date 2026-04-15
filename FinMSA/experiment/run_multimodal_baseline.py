"""
独立的多模态实验运行脚本，仅支持 TextCNNResNet 模型。
原 baseline 模型已被移除。
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn import metrics
from argparse import ArgumentParser

# 确保项目根目录在Python路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config, _MODEL_CLASSES
from data_utils.multimodal_data import get_multimodal_loaders
from models.multimodal_baseline import TextCNNResNet


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultimodalTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        # 类别权重（与原有逻辑一致）
        self.weights = torch.tensor([0.3, 0.4, 0.3]).to(self.device)

    def train(self, model, train_loader, val_loader, test_loader):
        criterion = nn.CrossEntropyLoss(weight=self.weights)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=getattr(self.config, 'weight_decay', 0.0)
        )

        best_val_f1 = 0.0
        best_epoch = 0
        best_state = None

        print(f"\n{'='*50}")
        print(f"开始训练 TextCNNResNet 模型")
        print(f"图像编码器: {getattr(self.config, 'image_encoder', 'resnet50')}")
        print(f"文本编码器: {self.config.encoder_type}")
        print(f"批次大小: {self.config.batch_size}")
        print(f"学习率: {self.config.lr}")
        print(f"{'='*50}\n")

        for epoch in range(self.config.epochs):
            model.train()
            total_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                labels = batch['labels'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                images = batch['images'].to(self.device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, token_type_ids, images)
                loss = criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 20 == 0:
                    print(f"Epoch[{epoch}] Batch[{batch_idx}] Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)

            val_acc, val_macf1, val_wf1 = self.evaluate(model, val_loader)
            print(f"\nEpoch[{epoch}] 完成 | Train Loss: {avg_loss:.4f}")
            print(f"Validation | Acc: {val_acc*100:.2f}% | Macro-F1: {val_macf1*100:.2f}% | Weight-F1: {val_wf1*100:.2f}%")

            if val_macf1 > best_val_f1:
                best_val_f1 = val_macf1
                best_epoch = epoch
                best_state = model.state_dict().copy()

                save_path = os.path.join(project_root, 'save_results', f'{self.config.save_name}_best.pt')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_state, save_path)
                print(f"✅ 保存最佳模型 (Macro-F1: {val_macf1*100:.2f}%)")

        print(f"\n{'='*50}")
        print(f"训练结束，加载最佳模型(Epoch {best_epoch})进行测试")
        print(f"{'='*50}")

        model.load_state_dict(best_state)
        test_acc, test_macf1, test_wf1 = self.evaluate(model, test_loader, verbose=True)

        print(f"\n🏆 最终结果:")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print(f"Test Macro-F1: {test_macf1*100:.2f}%")
        print(f"Test Weight-F1: {test_wf1*100:.2f}%")

        return test_acc, test_macf1, test_wf1

    def evaluate(self, model, data_loader, verbose=False):
        model.eval()
        y_pred = []
        y_label = []

        with torch.no_grad():
            for batch in data_loader:
                labels = batch['labels'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                images = batch['images'].to(self.device)

                logits = model(input_ids, attention_mask, token_type_ids, images)
                preds = torch.argmax(logits, dim=1)

                y_pred.append(preds.cpu().numpy())
                y_label.append(labels.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_label = np.concatenate(y_label)

        acc = metrics.accuracy_score(y_true=y_label, y_pred=y_pred)
        macf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='macro')
        wf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='weighted')

        if verbose:
            print("\n测试集混淆矩阵:")
            cm = metrics.confusion_matrix(y_true=y_label, y_pred=y_pred)
            print(cm)

        return acc, macf1, wf1


def main():
    parser = ArgumentParser(description="Multimodal Sentiment Analysis with TextCNNResNet")
    parser.add_argument('--config', default='config_dir/config_multimodal.ini',
                        help='配置文件路径（相对于项目根目录或绝对路径）')
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    print(f"正在加载配置文件: {config_path}")
    config = Config(config_path)

    # 设置默认值
    if not hasattr(config, 'seed'):
        config.seed = 42
    if not hasattr(config, 'device'):
        config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if not hasattr(config, 'drop_out'):
        config.drop_out = 0.5
    if not hasattr(config, 'num_classes'):
        config.num_classes = 3
    if not hasattr(config, 'max_grad_norm'):
        config.max_grad_norm = 1.0
    if not hasattr(config, 'weight_decay'):
        config.weight_decay = 0.0

    set_seed(config.seed)

    # 获取分词器
    encoder_info = _MODEL_CLASSES[config.encoder_type]
    tokenizer = encoder_info['tokenizer'].from_pretrained(
        getattr(config, f'{config.encoder_type}_path')
    )

    print("正在加载多模态数据...")
    train_loader, dev_loader, test_loader = get_multimodal_loaders(config, tokenizer)

    print("正在初始化 TextCNNResNet 模型...")
    model = TextCNNResNet(config).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    trainer = MultimodalTrainer(config)
    acc, macf1, wf1 = trainer.train(model, train_loader, dev_loader, test_loader)

    result_file = os.path.join(project_root, 'save_results', f'{config.save_name}_results.txt')
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w') as f:
        f.write(f"Model: TextCNNResNet\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Text Encoder: {config.encoder_type}\n")
        f.write(f"Image Encoder: {getattr(config, 'image_encoder', 'resnet50')}\n")
        f.write(f"Accuracy: {acc*100:.2f}%\n")
        f.write(f"Macro-F1: {macf1*100:.2f}%\n")
        f.write(f"Weighted-F1: {wf1*100:.2f}%\n")
    print(f"\n结果已保存至: {result_file}")

if __name__ == '__main__':
    main()