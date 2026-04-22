import torch
import torch.nn as nn
from transformers import BertModel, AutoModel, AutoConfig

class BERT_cls(nn.Module):
    def __init__(self, config, num_label=3, model=None, dropout=0.5, special_tokenizer=None):
        super(BERT_cls, self).__init__()
        self.model = model.from_pretrained(config.bert_path)
        if special_tokenizer is not None:
            self.model.resize_token_embeddings(len(special_tokenizer))

        # 从模型配置中获取隐藏层大小
        hidden_size = self.model.config.hidden_size

        self.activation = nn.Tanh()
        self.classifier = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # 使用 pooled output
        bert_emb = self.dropout(outputs[1])
        logits = self.classifier(bert_emb)
        return logits