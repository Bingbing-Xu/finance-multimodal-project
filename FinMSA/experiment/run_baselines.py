import sys
import os
# 获取当前文件所在目录的上级目录（即项目根目录）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ViTModel, ViltProcessor
from models.image_classifier import ImageClassifier
from data_utils.image_dataset import StockImageDataset
import os
import json
from sklearn import metrics
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_image_only(config):
    device = torch.device(config['device'])
    set_seed(config['seed'])
    print("1. 创建数据集...")
    train_dataset = StockImageDataset(config['data_dir'], 'train')
    val_dataset = StockImageDataset(config['data_dir'], 'dev')
    test_dataset = StockImageDataset(config['data_dir'], 'test')
    print("2. 数据集创建完成，创建 DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    print("3. DataLoader 创建完成，初始化模型...")
    model = ImageClassifier(num_classes=3, model_name=config['image_model']).to(device)
    print("4. 模型初始化完成，开始训练循环...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # 每 10 个 batch 打印一次
            if i % 10 == 0:
                print(f"Epoch {epoch}, batch {i}, loss: {loss.item():.4f}")
        # 验证
        val_acc, val_f1 = evaluate_image(model, val_loader, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"save_results/{config['save_name']}_best.pt")
    # 测试
    model.load_state_dict(torch.load(f"save_results/{config['save_name']}_best.pt"))
    test_acc, test_f1, test_wf1 = evaluate_image(model, test_loader, device, return_metrics=True)
    print(f"Test Acc: {test_acc:.4f}, Macro-F1: {test_f1:.4f}, Weighted-F1: {test_wf1:.4f}")


def evaluate_image(model, loader, device, return_metrics=False):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label']
            logits = model(pixel_values)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    acc = metrics.accuracy_score(all_labels, all_preds)
    f1_macro = metrics.f1_score(all_labels, all_preds, average='macro')
    f1_weighted = metrics.f1_score(all_labels, all_preds, average='weighted')
    if return_metrics:
        return acc, f1_macro, f1_weighted
    return acc, f1_macro


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True, choices=['image-only', 'vilt'])
    parser.add_argument('--config', type=str, help='Path to config JSON')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.baseline == 'image-only':
        train_image_only(config)
