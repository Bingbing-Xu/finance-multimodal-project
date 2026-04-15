import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.dataset_utils import normalize_word


class MultimodalStockDataset(Dataset):
    """
    完全独立的多模态数据集类，不修改原有text_dataset.py
    直接读取JSON并处理文本+图像
    """

    def __init__(self, config, split='train', tokenizer=None):
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.reserve_length = getattr(config, 'reserve_length', 180)

        # 路径配置
        data_root = config.root_dir
        self.image_dir = getattr(config, 'image_dir', os.path.join(data_root, 'images'))

        # 加载数据
        data_file = os.path.join(data_root, f'{split}.json')
        with open(data_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        # 消融实验参数（与原有配置兼容）
        self.use_text = getattr(config, 'use_text', True)
        self.use_description = getattr(config, 'use_description', True)

        # 图像预处理（直接缩放到224×224，避免小图裁剪错误）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 处理文本数据
        self.processed_data = self._process_data()

    def _process_data(self):
        """处理文本，与原有StockKnow逻辑保持一致"""
        processed = []

        for item in self.raw_data:
            # 标签处理：-1,0,1 -> 0,1,2
            label = int(item['gt_label']) + 1

            # 目标词替换（与原有逻辑一致）
            target = item['target']
            text = item['text'].replace('$T$', target)
            text = normalize_word(text)

            # 描述处理
            description = normalize_word(item.get('description', ''))
            if description:
                description = ' '.join(description.strip().split()[:self.reserve_length])

            # 构建输入文本（与StockKnow bert_cls_process一致）
            parts = [target]
            if self.use_text:
                parts.append(text)
            if self.use_description:
                parts.append(description)

            input_text = self.tokenizer.sep_token.join(parts)

            # 编码
            encoding = self.tokenizer(
                input_text,
                return_token_type_ids=True,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors='pt'
            )

            # 图像路径
            image_id = str(item['ImageID'])
            image_path = self._find_image_path(image_id)

            processed.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'token_type_ids': encoding['token_type_ids'].squeeze(0),
                'label': label,
                'image_path': image_path,
                'image_id': image_id
            })

        return processed

    def _find_image_path(self, image_id):
        """查找图像文件（支持多种扩展名）"""
        exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        for ext in exts:
            path = os.path.join(self.image_dir, image_id + ext)
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        # 加载图像
        if item['image_path'] and os.path.exists(item['image_path']):
            try:
                image = Image.open(item['image_path']).convert('RGB')
                image = self.transform(image)
            except Exception as e:
                print(f"Warning: 无法加载图像 {item['image_path']}, 使用零张量")
                image = torch.zeros(3, 224, 224)
        else:
            # 缺失图像用零张量代替
            image = torch.zeros(3, 224, 224)

        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'token_type_ids': item['token_type_ids'],
            'images': image,
            'labels': item['label']
        }


def multimodal_collate_fn(batch):
    """批次合并函数"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'token_type_ids': torch.stack([item['token_type_ids'] for item in batch]),
        'images': torch.stack([item['images'] for item in batch]),
        'labels': torch.tensor([item['labels'] for item in batch])
    }


def get_multimodal_loaders(config, tokenizer):
    """创建多模态数据加载器"""
    train_set = MultimodalStockDataset(config, 'train', tokenizer)
    dev_set = MultimodalStockDataset(config, 'dev', tokenizer)
    test_set = MultimodalStockDataset(config, 'test', tokenizer)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=multimodal_collate_fn,
        num_workers=0
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=0
    )

    print(f"多模态数据加载完成: Train[{len(train_set)}] Dev[{len(dev_set)}] Test[{len(test_set)}]")
    return train_loader, dev_loader, test_loader