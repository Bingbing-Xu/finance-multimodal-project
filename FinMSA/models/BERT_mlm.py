from transformers import BertForMaskedLM, RobertaForMaskedLM
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, RobertaForMaskedLM
import torch.nn as nn
import torch


class BERT_mlm(nn.Module):

    def __init__(self, config, rels_num=0, device='cpu', id2name=None, encoder=None, tokenizer=None,
                 init_by_cls=False):
        super(BERT_mlm, self).__init__()

        self.rels_num = rels_num
        self.device = device

        # 加载预训练模型
        self.encoder = encoder.from_pretrained(config.bert_path)
        self.config = config

        # 调整词表大小：原词表 + 1(replay token) + rels_num(情感标签token)
        original_vocab_size = self.encoder.config.vocab_size
        new_vocab_size = original_vocab_size + 1 + rels_num
        self.encoder.resize_token_embeddings(new_vocab_size)
        self.encoder.config.vocab_size = new_vocab_size  # 手动更新

        self.output_size = config.encoder_output_size
        self.tokenizer = tokenizer
        self.replay_token_id = torch.LongTensor([original_vocab_size]).to(device)

        if config.p_mask >= 0:
            self.p_mask = config.p_mask
            self.mlm_loss_fn = nn.CrossEntropyLoss()

        # 初始化新 token 的嵌入
        word_embeddings = self.encoder.get_input_embeddings()
        if id2name is not None:
            candidate_ids = tokenizer.batch_encode_plus(
                id2name,
                add_special_tokens=False,
                padding=False,
                truncation=False
            )['input_ids']

            std1 = torch.std(word_embeddings.weight.data[-rels_num:]).cpu().item()
            for i in range(rels_num):
                label_token_ids = candidate_ids[i]
                label_emb_init = word_embeddings.weight.data[label_token_ids].mean(dim=0)
                word_embeddings.weight.data[-rels_num:][i] = label_emb_init
            std2 = torch.std(word_embeddings.weight.data[-rels_num:]).cpu().item()
            print(f"Init new tokens by average, variance {std1:.4f} -> {std2:.4f}")
        else:
            print("Random init new tokens")

        self.mask_id = self.tokenizer.mask_token_id
        self.to(self.device)

    def forward(self, input_ids, attention_mask, token_type_ids, return_mask_hidden=False, return_cls_hidden=False):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=return_mask_hidden
        )

        x_idxs, y_idxs = torch.where(input_ids == self.mask_id)
        if x_idxs.shape[0] == 0:
            raise ValueError("No [MASK] token found in input")

        logits = out.logits[x_idxs, y_idxs]
        label_logits = logits[:, -self.rels_num:]

        if not return_mask_hidden:
            return label_logits
        else:
            last_hidden_states = out.hidden_states[-1]
            mask_hidden = last_hidden_states[x_idxs, y_idxs]
            if not return_cls_hidden:
                return label_logits, mask_hidden
            else:
                cls_hidden = last_hidden_states[:, 0, :]
                return label_logits, mask_hidden, cls_hidden

    def mask_replay_forward(self, input_ids, attention_mask, token_type_ids,
                            reason_input_ids, reason_attention_mask, reason_token_type_ids,
                            return_mask_hidden=False, return_cls_hidden=False):

        B, L = reason_input_ids.shape
        p_randn = torch.rand([B, L], device=self.device)
        x_mask, y_mask = torch.where((p_randn < self.p_mask) & (reason_token_type_ids == 1))

        if x_mask.shape[0] > 0:
            mask_reason_token_ids = reason_input_ids[x_mask, y_mask].clone()
            reason_input_ids = reason_input_ids.clone()
            reason_input_ids[x_mask, y_mask] = self.tokenizer.mask_token_id
        else:
            mask_reason_token_ids = torch.tensor([], device=self.device, dtype=torch.long)

        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=return_mask_hidden
        )
        reason_out = self.encoder(
            input_ids=reason_input_ids,
            attention_mask=reason_attention_mask
        )

        x_idxs, y_idxs = torch.where(input_ids == self.tokenizer.mask_token_id)
        logits = out.logits[x_idxs, y_idxs]

        if x_mask.shape[0] > 0:
            logits_mask = reason_out.logits[x_mask, y_mask]
            loss_mlm = self.mlm_loss_fn(logits_mask, mask_reason_token_ids)
        else:
            loss_mlm = torch.tensor(0.0, device=self.device)

        if torch.isnan(loss_mlm) or self.p_mask <= 0:
            loss_mlm = torch.tensor(0.0, device=self.device)

        label_logits = logits[:, -self.rels_num:]

        if not return_mask_hidden:
            return label_logits, loss_mlm
        else:
            last_hidden_states = out.hidden_states[-1]
            mask_hidden = last_hidden_states[x_idxs, y_idxs]
            if not return_cls_hidden:
                return label_logits, mask_hidden, loss_mlm
            else:
                cls_hidden = last_hidden_states[:, 0, :]
                return label_logits, mask_hidden, cls_hidden, loss_mlm

    def mlm_forward(self, mask_hidden):
        prediction_scores = self.encoder.cls(mask_hidden)
        return prediction_scores[:, -self.rels_num:]











