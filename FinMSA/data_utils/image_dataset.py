import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from torchvision import transforms


class StockImageDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        """
        data_dir: 包含 train.json, dev.json, test.json 的目录
        split: 'train', 'dev', 'test'
        """
        self.data_dir = data_dir
        with open(os.path.join(data_dir, f'{split}.json'), 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.img_root = os.path.join(data_dir, 'images')
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_id = str(item['ImageID'])
        # 尝试多种扩展名
        possible_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']
        img_path = None
        for ext in possible_extensions:
            test_path = os.path.join(self.img_root, f'{img_id}{ext}')
            if os.path.exists(test_path):
                img_path = test_path
                break
        if img_path is None:
            raise FileNotFoundError(
                f"Image for ID {img_id} not found in {self.img_root} with any extension {possible_extensions}")

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = int(item['gt_label']) + 1
        return {
            'pixel_values': image,
            'label': torch.tensor(label, dtype=torch.long)
        }