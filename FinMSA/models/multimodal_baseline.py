import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel, RobertaModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import _MODEL_CLASSES


class ImageEncoder(nn.Module):
    """图像编码器，支持ResNet和ViT，输出特征图（保留空间维度）"""

    def __init__(self, encoder_type='resnet50', freeze=False):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type.startswith('resnet'):
            if encoder_type == 'resnet50':
                base_model = models.resnet50(pretrained=True)
                self.feature_dim = base_model.fc.in_features  # 2048
            elif encoder_type == 'resnet34':
                base_model = models.resnet34(pretrained=True)
                self.feature_dim = base_model.fc.in_features  # 512
            elif encoder_type == 'resnet18':
                base_model = models.resnet18(pretrained=True)
                self.feature_dim = base_model.fc.in_features  # 512
            else:
                raise ValueError(f"Unsupported ResNet type: {encoder_type}")
            # 去掉最后的 avgpool 和 fc 层，保留特征图 [B, C, H, W]
            self.encoder = nn.Sequential(*list(base_model.children())[:-2])

        elif encoder_type == 'vit':
            base_model = models.vit_b_16(pretrained=True)
            self.feature_dim = 768
            self.encoder = base_model
            self.encoder.heads = nn.Identity()  # 移除分类头

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)                     # [B, C, H, W] for ResNet
        if self.encoder_type.startswith('resnet'):
            # 保持特征图，不做全局池化
            pass
        return features


class TextCNNResNet(nn.Module):
    """
    TextCNN + ResNet 多模态模型
    - 文本部分：使用预训练BERT的嵌入层，然后通过多个卷积核提取特征
    - 图像部分：ResNet编码器（输出全局特征）
    - 融合：拼接后经MLP分类
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---------- 文本编码器 ----------
        # 使用与BERT相同的嵌入层（可加载预训练权重）
        encoder_info = _MODEL_CLASSES[config.encoder_type]
        # 这里encoder_type应为'bert'或'roberta'等，我们借用其嵌入层
        self.text_embedding = encoder_info['model'].from_pretrained(
            getattr(config, f'{config.encoder_type}_path')
        ).embeddings.word_embeddings
        self.embedding_dim = self.text_embedding.embedding_dim

        # 可选：冻结嵌入层
        if getattr(config, 'freeze_text_encoder', False):
            for param in self.text_embedding.parameters():
                param.requires_grad = False

        # TextCNN参数
        filter_sizes = getattr(config, 'filter_sizes', '3,4,5')   # 卷积核尺寸，逗号分隔
        num_filters = getattr(config, 'num_filters', 100)         # 每种尺寸的卷积核数量
        self.filter_sizes = [int(ks) for ks in filter_sizes.split(',')]
        self.num_filters_total = num_filters * len(self.filter_sizes)

        # 定义多个卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (ks, self.embedding_dim))   # 输入通道=1（嵌入序列视为单通道图像）
            for ks in self.filter_sizes
        ])

        # ---------- 图像编码器 ----------
        image_encoder_type = getattr(config, 'image_encoder', 'resnet50')
        freeze_image = getattr(config, 'freeze_image_encoder', False)
        self.image_encoder = ImageEncoder(image_encoder_type, freeze=freeze_image)
        self.image_hidden_size = self.image_encoder.feature_dim
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))   # 用于图像特征全局池化

        # ---------- 融合层 ----------
        fusion_dim = self.num_filters_total + self.image_hidden_size
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.drop_out),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.drop_out)
        )
        self.classifier = nn.Linear(256, getattr(config, 'num_classes', 3))

    def forward(self, input_ids, attention_mask, token_type_ids, images):
        """
        input_ids: [batch, seq_len]   (已通过BERT tokenizer编码)
        attention_mask: [batch, seq_len]  (用于指示有效token，TextCNN中可用作后续mask)
        token_type_ids: [batch, seq_len]  (此处忽略)
        images: [batch, 3, 224, 224]
        """
        # ---------- 文本特征 ----------
        # 获取词嵌入 [batch, seq_len, embed_dim]
        embedded = self.text_embedding(input_ids)

        # 应用attention mask：将padding位置的嵌入置零（可选）
        if attention_mask is not None:
            embedded = embedded * attention_mask.unsqueeze(-1).float()

        # 增加通道维度，变为 [batch, 1, seq_len, embed_dim]
        embedded = embedded.unsqueeze(1)

        # 卷积 + 池化
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # 每个卷积输出 [batch, num_filters, seq_len-ks+1]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conved]     # 每个输出 [batch, num_filters]
        text_features = torch.cat(pooled, dim=1)                              # [batch, num_filters_total]

        # ---------- 图像特征 ----------
        image_features = self.image_encoder(images)            # [batch, C, H, W]
        image_features = self.global_avg_pool(image_features)  # [batch, C, 1, 1]
        image_features = image_features.view(image_features.size(0), -1)  # [batch, C]

        # ---------- 融合 ----------
        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.fusion_layer(combined)
        logits = self.classifier(fused)
        return logits