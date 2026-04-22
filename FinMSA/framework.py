
import time  # 确保单独导入time
import datetime as dt  # 给datetime取别名
import torch
import torch.nn as nn
import numpy as np
import logging
from sklearn import metrics

class CLS_framework(object):
    def __init__(self, config):
        self.config = config
        self.average = config.average
        # 这里本来想控制不同类别样本的损失权重，发现没什么用，如果需要可以在损失函数声明时加入
        self.weights = torch.tensor([0.3, 0.4, 0.3]).to(config.device)

    def train(self, config, model, train_loader, val_loader, test_loader):
        loss_function = nn.CrossEntropyLoss(weight=self.weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)

        best_val_f1 = 0.0
        best_epoch = 0
        best_model_state = None

        for epoch in range(config.epochs):
            print(f"\n=== Epoch {epoch} ===")
            model.train()
            total_loss = 0.0
            total_samples = 0

            for labels, input_ids, attention_mask, token_type_ids in train_loader:
                # 数据移至设备
                labels = labels.to(config.device)
                input_ids = input_ids.to(config.device)
                attention_mask = attention_mask.to(config.device)
                token_type_ids = token_type_ids.to(config.device)

                # 模型预测
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = loss_function(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_samples += logits.size(0)

            avg_train_loss = total_loss / len(train_loader)  # 平均每个batch的loss

            # ---------- 验证 ----------
            val_acc, val_macf1, val_wf1 = self.valid(config, model, val_loader)
            print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Acc: {val_acc * 100:.2f}% | Val Macro-F1: {val_macf1 * 100:.2f}% | "
                  f"Val Weighted-F1: {val_wf1 * 100:.2f}%")

            # 保存最佳模型（基于验证集 Macro F1）
            if val_macf1 > best_val_f1:
                best_val_f1 = val_macf1
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, f'./save_results/{config.save_name}_best.pt')
                print(f"✅ 保存最佳模型，Epoch {epoch} (Val Macro-F1 = {val_macf1 * 100:.2f}%)")

        print("\n=== 训练完成===")
        print(f"最佳模型在验证集上的 Macro-F1: {best_val_f1 * 100:.2f}% (Epoch {best_epoch})")
        return best_val_f1, best_epoch

    def valid(self, config, model, val_loader):
        y_pred = []
        y_label = []
        model.eval()
        with torch.no_grad():
            for labels, input_ids, attention_mask, token_type_ids in val_loader:
                # 选取对应批次数据的输入和标签
                labels, input_ids, token_type_ids, attention_mask = labels.to(config.device), input_ids.to(
                    config.device), attention_mask.to(config.device), token_type_ids.to(config.device)

                # 模型预测
                logits = model(input_ids, attention_mask, token_type_ids)

                # 取分类概率最大的类别作为预测的类别
                y_hat = torch.tensor([torch.argmax(_) for _ in logits]).to(config.device)

                y_pred.append(y_hat.cpu().numpy())
                y_label.append(labels.cpu().numpy())

            y_pred = np.concatenate(y_pred)
            y_label = np.concatenate(y_label)

            acc = metrics.accuracy_score(y_true=y_label, y_pred=y_pred)
            macf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='macro')
            wf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='weighted')
            confusion_matrix = metrics.confusion_matrix(y_true=y_label, y_pred=y_pred)
            print("Confusion matrix:")
            print(confusion_matrix)

            return acc, macf1, wf1

    def test(self, config, model, test_loader, ckpt=None):
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt))
        results = self.valid(config, model, test_loader)
        return results

#运行main调用的train是这个
class MLM_framework(object):
    def __init__(self, config):
        self.config = config
        self.average = config.average
        # 这里本来想控制不同类别样本的损失权重，发现没什么用，如果需要可以在损失函数声明时加入
        self.weights = torch.tensor([0.3, 0.4, 0.3]).to(config.device)

    def train(self, config, model, train_loader, val_loader, test_loader):
        print("=== 开始训练 (MLM 框架) ===")
        loss_function = nn.CrossEntropyLoss(weight=self.weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)

        best_val_f1 = 0.0
        best_epoch = 0
        best_model_state = None

        for epoch in range(config.epochs):
            print(f"\n=== Epoch {epoch} ===")
            model.train()
            total_loss = 0.0

            for i, (labels, input_ids, attention_mask, token_type_ids) in enumerate(train_loader):
                # 数据移至设备（只移动一次，删除重复代码）
                labels = labels.to(config.device)
                input_ids = input_ids.to(config.device)
                attention_mask = attention_mask.to(config.device)
                token_type_ids = token_type_ids.to(config.device)

                # 模型前向，需要 return_mask_hidden=True
                logits, mask_hidden = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    return_mask_hidden=True
                )

                loss = loss_function(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # ---------- 验证 ----------
            val_acc, val_macf1, val_wf1 = self.valid(config, model, val_loader)
            print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Acc: {val_acc * 100:.2f}% | Val Macro-F1: {val_macf1 * 100:.2f}% | "
                  f"Val Weighted-F1: {val_wf1 * 100:.2f}%")

            # 保存最佳模型（基于验证集 Macro F1）
            if val_macf1 > best_val_f1:
                best_val_f1 = val_macf1
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, f'./save_results/{config.save_name}_best.pt')
                print(f"✅ 保存最佳模型，Epoch {epoch} (Val Macro-F1 = {val_macf1 * 100:.2f}%)")

        print("\n=== 训练完成===")
        print(f"最佳模型在验证集上的 Macro-F1: {best_val_f1 * 100:.2f}% (Epoch {best_epoch})")

        return best_val_f1, best_epoch

    def valid(self, config, model, val_loader):
        print("进入valid")
        y_pred = []
        y_label = []
        model.eval()
        with torch.no_grad():
            for labels, input_ids, attention_mask, token_type_ids in val_loader:
                # 选取对应批次数据的输入和标签
                labels, input_ids, token_type_ids, attention_mask = labels.to(config.device), input_ids.to(
                    config.device), attention_mask.to(config.device), token_type_ids.to(config.device)

                # 模型预测
                logits, mask_hidden = model(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                  attention_mask=attention_mask,
                                                  return_mask_hidden=True)

                # 取分类概率最大的类别作为预测的类别
                y_hat = torch.tensor([torch.argmax(_) for _ in logits]).to(config.device)
                #print("预测类别完成")

                y_pred.append(y_hat.cpu().numpy())
                y_label.append(labels.cpu().numpy())

            y_pred = np.concatenate(y_pred)
            y_label = np.concatenate(y_label)

            print("计算指标")

            acc = metrics.accuracy_score(y_true=y_label, y_pred=y_pred)
            # p = metrics.precision_score(y_true=y_label, y_pred=y_pred, average=self.average)
            # r = metrics.recall_score(y_true=y_label, y_pred=y_pred, average=self.average)
            macf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='macro')
            wf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='weighted')
            confusion_matrix = metrics.confusion_matrix(y_true=y_label, y_pred=y_pred)
            print("Confusion matrix:")
            print(confusion_matrix)

            return acc, macf1, wf1

    def test(self, config, model, test_loader, ckpt=None):
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt))
        results = self.valid(config, model, test_loader)
        return results

class MLM_plus_framework(object):
    def __init__(self, config):
        self.config = config
        self.average = config.average
        # 这里本来想控制不同类别样本的损失权重，发现没什么用，如果需要可以在损失函数声明时加入
        self.weights = torch.tensor([0.3, 0.4, 0.3]).to(config.device)

    def train(self, config, model, train_loader, val_loader, test_loader):
        print("=== 开始训练 (MLM+ 框架) ===")
        loss_function = nn.CrossEntropyLoss(weight=self.weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)

        best_val_f1 = 0.0
        best_epoch = 0
        best_model_state = None

        for epoch in range(config.epochs):
            print(f"\n=== Epoch {epoch} ===")
            model.train()
            total_loss = 0.0
            total_samples = 0

            for i, (labels, input_ids, attention_mask, token_type_ids,
                    reason_input_ids, reason_attention_mask, reason_token_type_ids) in enumerate(train_loader):
                #print(f"Batch {i}: 进入二层循环")
                # 数据移至设备
                labels = labels.to(config.device)
                input_ids = input_ids.to(config.device)
                attention_mask = attention_mask.to(config.device)
                token_type_ids = token_type_ids.to(config.device)
                reason_input_ids = reason_input_ids.to(config.device)
                reason_attention_mask = reason_attention_mask.to(config.device)
                reason_token_type_ids = reason_token_type_ids.to(config.device)

                # 前向传播（带MLM replay）
                logits, mask_hidden, mlm_loss = model.mask_replay_forward(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    reason_input_ids=reason_input_ids,
                    reason_token_type_ids=reason_token_type_ids,
                    reason_attention_mask=reason_attention_mask,
                    return_mask_hidden=True
                )

                loss = loss_function(logits, labels) + config.alpha * mlm_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_samples += logits.size(0)

            avg_train_loss = total_loss / (total_samples / config.batch_size)

            # ---------- 验证阶段 ----------
            val_acc, val_macf1, val_wf1 = self.valid(config, model, val_loader)
            print(
                f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | Val Macro-F1: {val_macf1 * 100:.2f}% | Val Weighted-F1: {val_wf1 * 100:.2f}%")

            # 保存最佳模型（基于验证集 Macro F1）
            if val_macf1 > best_val_f1:
                best_val_f1 = val_macf1
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, f'./save_results/{config.save_name}_best.pt')
                print(f"✅ 保存最佳模型，Epoch {epoch} (Val Macro-F1 = {val_macf1 * 100:.2f}%)")

        # ---------- 训练结束，测试最佳模型 ----------
        print("\n=== 训练完成===")
        print(f"最佳模型在验证集上的 Macro-F1: {best_val_f1 * 100:.2f}% (Epoch {best_epoch})")

        return best_val_f1, best_epoch

    def valid(self, config, model, val_loader):
        print("进入valid")
        y_pred = []
        y_label = []
        model.eval()
        with torch.no_grad():
            for labels, input_ids, attention_mask, token_type_ids, _, _, _ in val_loader:
                # 选取对应批次数据的输入和标签
                labels, input_ids, token_type_ids, attention_mask = labels.to(config.device), input_ids.to(
                    config.device), attention_mask.to(config.device), token_type_ids.to(config.device)

                # 模型预测
                logits, mask_hidden = model(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                  attention_mask=attention_mask,
                                                  return_mask_hidden=True)

                # 取分类概率最大的类别作为预测的类别
                y_hat = torch.tensor([torch.argmax(_) for _ in logits]).to(config.device)

                y_pred.append(y_hat.cpu().numpy())
                y_label.append(labels.cpu().numpy())

            y_pred = np.concatenate(y_pred)
            y_label = np.concatenate(y_label)

            print("计算指标")
            acc = metrics.accuracy_score(y_true=y_label, y_pred=y_pred)

            macf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='macro')
            wf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='weighted')
            confusion_matrix = metrics.confusion_matrix(y_true=y_label, y_pred=y_pred)
            print("Confusion matrix:")
            print(confusion_matrix)

            return acc, macf1, wf1

    def test(self, config, model, test_loader, ckpt=None):
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt))
        results = self.valid(config, model, test_loader)
        return results
