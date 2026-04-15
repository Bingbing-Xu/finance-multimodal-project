import json
import torch
from torch.utils.data import Dataset
import os

class TextOnlyDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_length=400):
        """
        data_dir: 包含 train.json, dev.json, test.json 的目录
        split: 'train', 'dev', 'test'
        tokenizer: HuggingFace tokenizer
        max_length: 最大序列长度
        """
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载 JSON 文件
        json_path = os.path.join(data_dir, f'{split}.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        # 将 gt_label 从字符串转为整数，并映射到 0,1,2（假设原始为 -1,0,1）
        label = int(item['gt_label']) + 1

        # 使用 tokenizer 编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),      # [seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }