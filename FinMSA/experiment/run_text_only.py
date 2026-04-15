import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn import metrics
import numpy as np
import random

# 导入自定义数据集
from experiment.text_dataset import TextOnlyDataset

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 强制使用确定性卷积算法
    torch.backends.cudnn.benchmark = False  # 禁止自动寻找最优算法（避免非确定性优

def train_text_only(config):
    device = torch.device(config['device'])
    set_seed(config['seed'])

    # 初始化 tokenizer 和模型
    print("1. 初始化 tokenizer 和模型...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=3
    ).to(device)

    print("2. 创建数据集...")
    train_dataset = TextOnlyDataset(
        config['data_dir'], 'train', tokenizer, max_length=config.get('max_length', 400)
    )
    val_dataset = TextOnlyDataset(
        config['data_dir'], 'dev', tokenizer, max_length=config.get('max_length', 400)
    )
    test_dataset = TextOnlyDataset(
        config['data_dir'], 'test', tokenizer, max_length=config.get('max_length', 400)
    )

    print("3. 创建 DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_epoch = 0
    best_model_state = None

    print("4. 开始训练...")
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        print(f"\nEpoch {epoch} 开始...")
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:
                print(f"  batch {i}, loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # 验证
        val_acc, val_macf1, val_wf1 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:2d} 完成 | Train Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macf1:.4f} | Val Weighted-F1: {val_wf1:.4f}")

        # 保存最佳模型（基于验证 Macro F1）
        if val_macf1 > best_val_f1:
            best_val_f1 = val_macf1
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"save_results/{config['save_name']}_best.pt")
            print(f"✅ 保存最佳模型，Epoch {epoch} (Val Macro-F1 = {val_macf1:.4f})")

    # 测试最佳模型
    print("\n=== 训练完成，加载最佳模型进行测试 ===")
    model.load_state_dict(best_model_state)
    test_acc, test_macf1, test_wf1 = evaluate(model, test_loader, device)
    print(f"\n🏆 最佳模型来自 Epoch {best_epoch}")
    print(f"Test Acc: {test_acc:.4f} | Test Macro-F1: {test_macf1:.4f} | Test Weighted-F1: {test_wf1:.4f}")

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = metrics.accuracy_score(all_labels, all_preds)
    macf1 = metrics.f1_score(all_labels, all_preds, average='macro')
    wf1 = metrics.f1_score(all_labels, all_preds, average='weighted')
    return acc, macf1, wf1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train_text_only(config)