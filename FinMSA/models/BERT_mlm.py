from transformers import BertForMaskedLM, RobertaForMaskedLM
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, RobertaForMaskedLM
import torch.nn as nn
import torch


class BERT_mlm(nn.Module):
    """def __init__(self, config, rels_num=0, device='cpu', id2name=None, encoder=None, tokenizer=None, init_by_cls=False):
        super(BERT_mlm, self).__init__()

        self.rels_num = rels_num
        self.device = device

        # 加载预训练 MLM 模型（兼容所有编码器）
        self.encoder = encoder.from_pretrained(config.bert_path)
        self.config = config

        # 记录原始词表大小，用于计算 replay token 索引
        original_vocab_size = self.encoder.config.vocab_size
        new_vocab_size = original_vocab_size + 1 + rels_num
        self.encoder.resize_token_embeddings(new_vocab_size)
        # 可选：更新配置中的 vocab_size（resize 已自动更新，此处为安全保留）
        self.encoder.config.vocab_size = new_vocab_size

        self.output_size = config.encoder_output_size
        self.tokenizer = tokenizer
        # replay token 是新增的第一个 token，索引为 original_vocab_size
        self.replay_token_id = torch.LongTensor([original_vocab_size]).to(device)

        if config.p_mask >= 0:
            self.p_mask = config.p_mask
            self.mlm_loss_fn = nn.CrossEntropyLoss()

        # ---------- 新 token 嵌入初始化 ----------
        word_embeddings = self.encoder.get_input_embeddings()

        if id2name is not None:
            # 将情感标签描述转为 token ids
            candidate_ids = tokenizer.batch_encode_plus(
                id2name,
                add_special_tokens=False,
                padding=False,
                truncation=False
            )['input_ids']

            std1 = torch.std(word_embeddings.weight.data[-rels_num:]).cpu().item()

            if init_by_cls:
                # 预留注意力加权平均接口，当前实现仍为简单平均
                # 如需实现真正的加权平均，可在此扩展
                print("Note: init_by_cls=True uses simple average (weighted attention not implemented).")
                for i in range(rels_num):
                    label_token_ids = candidate_ids[i]
                    label_emb_init = word_embeddings.weight.data[label_token_ids].mean(dim=0)
                    word_embeddings.weight.data[-rels_num:][i] = label_emb_init
            else:
                # 简单平均初始化
                for i in range(rels_num):
                    label_token_ids = candidate_ids[i]
                    label_emb_init = word_embeddings.weight.data[label_token_ids].mean(dim=0)
                    word_embeddings.weight.data[-rels_num:][i] = label_emb_init

            std2 = torch.std(word_embeddings.weight.data[-rels_num:]).cpu().item()
            print(f"Init new tokens by {'average' if not init_by_cls else 'attention-weighted-average (simplified)'}. "
                  f"Variance: {std1:.4f} -> {std2:.4f}")
        else:
            print("Random init the new tokens embedding.")

        self.mask_id = self.tokenizer.mask_token_id
        self.to(self.device)

    def forward(self, input_ids, attention_mask, token_type_ids, return_mask_hidden=False, return_cls_hidden=False):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=return_mask_hidden
        )

        # 定位 [MASK] 位置
        x_idxs, y_idxs = torch.where(input_ids == self.mask_id)
        if x_idxs.shape[0] == 0:
            raise ValueError("No [MASK] token found in input")

        # 取出 [MASK] 对应的 logits
        logits = out.logits[x_idxs, y_idxs]  # [num_mask, vocab_size]
        label_logits = logits[:, -self.rels_num:]  # 取最后 rels_num 个作为情感标签 logits

        if not return_mask_hidden:
            return label_logits
        else:
            last_hidden_states = out.hidden_states[-1]
            mask_hidden = last_hidden_states[x_idxs, y_idxs]  # [num_mask, hidden_size]
            if not return_cls_hidden:
                return label_logits, mask_hidden
            else:
                cls_hidden = last_hidden_states[:, 0, :]  # [batch, hidden_size]
                return label_logits, mask_hidden, cls_hidden

    def mask_replay_forward(self, input_ids, attention_mask, token_type_ids,
                            reason_input_ids, reason_attention_mask, reason_token_type_ids,
                            return_mask_hidden=False, return_cls_hidden=False):
        B, L = reason_input_ids.shape
        p_randn = torch.rand([B, L], device=self.device)
        x_mask, y_mask = torch.where((p_randn < self.p_mask) & (reason_token_type_ids == 1))

        # 克隆 reason_input_ids 以避免副作用
        reason_input_ids = reason_input_ids.clone()
        if x_mask.shape[0] > 0:
            mask_reason_token_ids = reason_input_ids[x_mask, y_mask].clone()
            reason_input_ids[x_mask, y_mask] = self.tokenizer.mask_token_id
        else:
            mask_reason_token_ids = torch.tensor([], device=self.device, dtype=torch.long)

        # 主输入前向
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=return_mask_hidden
        )
        # 推理文本前向（用于 MLM 损失）
        reason_out = self.encoder(
            input_ids=reason_input_ids,
            attention_mask=reason_attention_mask,
            token_type_ids=reason_token_type_ids,
            output_hidden_states=return_mask_hidden
        )

        # 主输入中的 [MASK] 位置
        x_idxs, y_idxs = torch.where(input_ids == self.tokenizer.mask_token_id)
        logits = out.logits[x_idxs, y_idxs]

        # 计算 MLM 损失（仅在有被 mask 的 token 时）
        if x_mask.shape[0] > 0:
            logits_mask = reason_out.logits[x_mask, y_mask]
            loss_mlm = self.mlm_loss_fn(logits_mask, mask_reason_token_ids)
        else:
            loss_mlm = torch.tensor(0.0, device=self.device)

        # 处理 NaN 或无效情况
        if loss_mlm.isnan().any() or self.p_mask <= 0:
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
        # 通过 MLM 头得到全词表 logits，再取最后 rels_num 个
        prediction_scores = self.encoder.cls(mask_hidden)
        return prediction_scores[:, -self.rels_num:]"""

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

    """def __init__(self, config, rels_num=0, device='cpu', id2name=None, encoder=None, tokenizer=None, init_by_cls=False):
        super(BERT_mlm, self).__init__()

        self.rels_num = rels_num
        self.device = device

        # 加载预训练模型（兼容所有编码器）
        self.encoder = encoder.from_pretrained(config.bert_path)
        self.config = config

        # 调整词表大小：原词表 + 1(replay) + rels_num(新token)
        self.encoder.resize_token_embeddings(self.encoder.config.vocab_size + 1 + rels_num)
        self.output_size = config.encoder_output_size
        self.tokenizer = tokenizer
        self.replay_token_id = torch.LongTensor([self.encoder.config.vocab_size - rels_num]).to(device)

        if config.p_mask >= 0:
            self.p_mask = config.p_mask
            self.mlm_loss_fn = nn.CrossEntropyLoss()

        # ---------- 新 token 嵌入初始化（统一处理）----------
        # 获取词嵌入层（通用接口）
        word_embeddings = self.encoder.get_input_embeddings()

        if id2name is not None:
            candidate_labels = id2name
            # 将候选标签转换为 token IDs
            candidate_ids = tokenizer.batch_encode_plus(
                candidate_labels,
                add_special_tokens=False,
                padding=False,
                truncation=False
            )['input_ids']

            # 初始化前的标准差（仅用于日志）
            std1 = torch.std(word_embeddings.weight.data[-rels_num:]).cpu().item()

            if init_by_cls:
                # 使用注意力加权平均（需要模型前向计算）
                for i in range(rels_num):
                    label_token_ids = candidate_ids[i]
                    # 将输入转为 tensor
                    input_tensor = torch.tensor([label_token_ids]).to(self.device)
                    # 前向传播获取 attention
                    outputs = self.encoder(
                        input_ids=input_tensor,
                        output_attentions=True,
                        return_dict=True
                    )
                    # 取最后一层 attention 的平均值作为权重（简化）
                    attentions = outputs.attentions  # tuple of (batch, head, seq, seq)
                    # 此处简化实现：使用 token embedding 的平均值（而非加权）
                    label_emb_init = word_embeddings.weight.data[label_token_ids].mean(dim=0)
                    word_embeddings.weight.data[-rels_num:][i] = label_emb_init
            else:
                # 简单平均
                for i in range(rels_num):
                    label_token_ids = candidate_ids[i]
                    label_emb_init = word_embeddings.weight.data[label_token_ids].mean(dim=0)
                    word_embeddings.weight.data[-rels_num:][i] = label_emb_init

            std2 = torch.std(word_embeddings.weight.data[-rels_num:]).cpu().item()
            print(
                f"Init the new tokens embedding by {'attention-weighted-average' if init_by_cls else 'average'} "
                f"of labels token-emb. The variance from {std1:.4f} to {std2:.4f}"
            )
        else:
            print("Random init the new tokens embedding.")

        self.mask_id = self.tokenizer.mask_token_id
        self.to(self.device)

    def forward(self, input_ids, attention_mask, token_type_ids, return_mask_hidden=False, return_cls_hidden=False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, output_hidden_states=return_mask_hidden)

        ## [MASK] position

        x_idxs, y_idxs = torch.where(input_ids == self.mask_id)
        # x_idxs, y_idxs = torch.where(input_ids == 103)

        ## [B, 30552+rels_num]
        logits = out.logits[x_idxs, y_idxs]

        assert logits.shape[0] == out.logits.shape[0]

        if not return_mask_hidden:
            # [B,rels_num]
            return logits[:, -self.rels_num:]
        else:
            # [B,L,768]
            last_hidden_states = out.hidden_states[-1]
            ## [B,768]
            mask_hidden = last_hidden_states[x_idxs, y_idxs]
            assert mask_hidden.shape[0] == last_hidden_states.shape[0]
            ## [B,rels_num], [B,768]
            if not return_cls_hidden:
                return logits[:, -self.rels_num:], mask_hidden
            else:
                cls_hidden = last_hidden_states[:, 0, :]
                return logits[:, -self.rels_num:], mask_hidden, cls_hidden

    def mask_replay_forward(self, input_ids, attention_mask, token_type_ids, reason_input_ids, reason_attention_mask,
                            reason_token_type_ids, return_mask_hidden=False,
                            return_cls_hidden=False):
        B = reason_input_ids.shape[0]
        L = reason_input_ids.shape[1]
        p_randn = torch.rand([B, L]).to(self.device)
        x_mask, y_mask = torch.where((p_randn < self.p_mask) * (reason_token_type_ids == 1))
        mask_reason_token_ids = reason_input_ids[x_mask, y_mask]
        reason_input_ids[x_mask, y_mask] = self.tokenizer.mask_token_id

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, output_hidden_states=return_mask_hidden)
        reason_out = self.encoder(input_ids=reason_input_ids, attention_mask=reason_attention_mask,
                                  output_hidden_states=return_mask_hidden)

        ## [MASK] position
        x_idxs, y_idxs = torch.where((input_ids == self.tokenizer.mask_token_id))

        ## [B, 30552+rels_num]
        logits = out.logits[x_idxs, y_idxs]
        logits_mask = reason_out.logits[x_mask, y_mask]
        loss_mlm = self.mlm_loss_fn(logits_mask, mask_reason_token_ids)

        if loss_mlm.cpu().isnan() or self.p_mask <= 0:
            loss_mlm = torch.zeros([1])[0].to(self.device)

        assert logits.shape[0] == out.logits.shape[0]

        if not return_mask_hidden:
            # [B,rels_num]
            return logits[:, -self.rels_num:], loss_mlm
        else:
            # [B,L,768]
            last_hidden_states = out.hidden_states[-1]
            ## [B,768]
            mask_hidden = last_hidden_states[x_idxs, y_idxs]
            assert mask_hidden.shape[0] == last_hidden_states.shape[0]
            ## [B,rels_num], [B,768]
            if not return_cls_hidden:
                return logits[:, -self.rels_num:], mask_hidden, loss_mlm
            else:
                cls_hidden = last_hidden_states[:, 0, :]
                return logits[:, -self.rels_num:], mask_hidden, cls_hidden, loss_mlm

    def mlm_forward(self, mask_hidden):
        # [B,30552+rels_num]
        prediction_scores = self.encoder.cls(mask_hidden)
        # [B,rels_num]
        return prediction_scores[:, -self.rels_num:]"""











