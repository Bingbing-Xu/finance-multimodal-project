import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel
import os


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=3, model_name='vit-base-patch16-224', pretrained=True):
        super().__init__()
        self.model_name = model_name

        if 'vit' in model_name:
            # 构建本地模型路径（假设模型存放在 pretrained_models/vit_cache/{model_name} 下）
            local_model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "pretrained_models", "vit_cache", model_name
            )

            # 检查本地路径是否存在
            if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
                # 从本地加载模型
                print(f"Loading ViT model from local path: {local_model_path}")
                self.backbone = ViTModel.from_pretrained(local_model_path)
            else:
                # 如果本地不存在，则尝试从在线下载（需配置代理）或直接报错
                # 这里我们选择报错，因为您已明确要求使用本地模型
                raise FileNotFoundError(
                    f"Local ViT model not found at {local_model_path}. "
                    "Please ensure the model is downloaded to this directory."
                )

            hidden_size = self.backbone.config.hidden_size
            self.use_vit = True

        elif model_name == 'resnet50':
            # 使用 torchvision ResNet50
            resnet = models.resnet50(pretrained=pretrained)
            resnet.fc = nn.Identity()
            self.backbone = resnet
            hidden_size = 2048
            self.use_vit = False

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        if self.use_vit:
            outputs = self.backbone(pixel_values=pixel_values)
            features = outputs.pooler_output
        else:
            features = self.backbone(pixel_values)
        logits = self.classifier(features)
        return logits