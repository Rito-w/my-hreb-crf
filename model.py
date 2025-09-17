from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torch import nn
import torch
from torchcrf import CRF
from mega_pytorch import MegaLayer
import os
import inspect
from typing import Optional


class ReducedBiasResidual(nn.Module):
    """
    动态残差门控：在跳连分支 x 与变换分支 y 之间进行特征维度的自适应加权。
    生成逐特征门控 alpha（作用于 y）与 beta（作用于 x）：
      out = sigma(Wa [x;y]) * y + sigma(Wb [x;y]) * x
    其中 [x;y] 表示在隐藏维上的拼接。
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj_alpha = nn.Linear(hidden_size * 2, hidden_size)
        self.proj_beta = nn.Linear(hidden_size * 2, hidden_size)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x、y 张量形状: (B, T, H)
        z = torch.cat([x, y], dim=-1)
        alpha = self.act(self.proj_alpha(z))  # 形状: (B, T, H)
        beta = self.act(self.proj_beta(z))   # 形状: (B, T, H)
        return alpha * y + beta * x


class HREBCRF(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, paper_aligned: Optional[bool] = None):
        super().__init__(config)
        self.num_labels = config.num_labels
        # 默认采用“论文对齐”行为，除非显式关闭
        self.paper_aligned = True if paper_aligned is None else bool(paper_aligned)

        self.r_lstm = nn.Parameter(torch.tensor(0.5))
        self.r_mega = nn.Parameter(torch.tensor(0.5))

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.5)
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        # 允许通过环境变量启用拉普拉斯注意力（更贴近论文的试验设置）
        def _parse_bool_env(name: str, default: bool = False) -> bool:
            val = os.environ.get(name)
            if val is None:
                return default
            return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

        use_laplacian = _parse_bool_env("MEGA_LAPLACIAN", False)
        self.mega = MegaLayer(
            dim=config.hidden_size,
            ema_heads=32,
            attn_dim_qk=128,
            attn_dim_value=256,
            laplacian_attn_fn=use_laplacian,
        )
        # 预先缓存 MegaLayer 接受的掩码参数名，避免每步调用时 try/except 的开销
        try:
            sig = inspect.signature(self.mega.forward)
            if 'mask' in sig.parameters:
                self._mega_mask_kw = 'mask'
            elif 'attention_mask' in sig.parameters:
                self._mega_mask_kw = 'attention_mask'
            else:
                self._mega_mask_kw = None
        except Exception:
            self._mega_mask_kw = None

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

        # 可选的“降低偏置”残差：在残差连接上引入动态 alpha/beta 门控
        self._reduced_bias_enabled = os.environ.get("REDUCED_BIAS", "0").strip().lower() in ("1", "true", "yes", "y", "on")
        if self._reduced_bias_enabled:
            self.rb_resid = ReducedBiasResidual(config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 最后一层隐藏状态
        bert_output = outputs[0]
        bert_output = self.dropout(bert_output)

        # LSTM 分支
        lstm_output, _ = self.bilstm(bert_output)

        # Mega 分支
        # 论文对齐：不向 MegaLayer 传递 attention_mask
        # 增强模式：若非论文对齐且支持掩码参数，则传入布尔掩码
        if self.paper_aligned:
            mega_output = self.mega(bert_output)
            r_lstm = self.r_lstm
            r_mega = self.r_mega
        else:
            if self._mega_mask_kw is not None and attention_mask is not None:
                mega_output = self.mega(bert_output, **{self._mega_mask_kw: attention_mask.bool()})
            else:
                mega_output = self.mega(bert_output)
            # 通过 sigmoid 将 r 压到 [0,1]，以提升稳定性
            r_lstm = torch.sigmoid(self.r_lstm)
            r_mega = torch.sigmoid(self.r_mega)

        lstm_output = r_lstm * lstm_output + (1 - r_lstm) * bert_output
        mega_output = r_mega * mega_output + (1 - r_mega) * bert_output

        merged = mega_output + lstm_output
        if getattr(self, "_reduced_bias_enabled", False):
            fused = self.rb_resid(bert_output, merged)
            merge_output = self.layer_norm(fused)
        else:
            merge_output = self.layer_norm(merged)

        logits = self.classifier(merge_output)

        # 使用 CRF：结合掩码，并对填充位进行安全处理
        loss = None
        mask = None
        paths = None
        offset = 0  # 当排除特殊符号（如 [CLS]）时的索引偏移
        if labels is not None:
            # 将被忽略的标签（-100）置为 0，确保为 long 类型
            labels_pad = labels.masked_fill(labels == -100, 0).to(dtype=torch.long)
            # 基于 labels/attention_mask 构造掩码：label != -100 且（如有）注意力为 1 的位置为有效
            if attention_mask is not None:
                base_mask = attention_mask.bool() & (labels != -100)
            else:
                base_mask = (labels != -100)

            if self.paper_aligned:
                mask = base_mask
                # 满足 torchcrf 约束：每条序列的首个时间步必须为 True
                if mask.dim() == 2:
                    # 极小化修复：若整条序列无有效位，则将首个真实 token 位置置为 True
                    no_valid = mask.sum(dim=1) == 0
                    if no_valid.any():
                        if attention_mask is not None:
                            first_idx = attention_mask.bool().float().argmax(dim=1)
                        else:
                            first_idx = torch.zeros(mask.size(0), dtype=torch.long, device=mask.device)
                        mask[torch.arange(mask.size(0), device=mask.device), first_idx] = True
                    # 严格约束：时间步 0 必须为 True
                    mask[:, 0] = True
                log_likelihood = self.crf(logits, labels_pad, mask=mask, reduction="mean")
                loss = 0 - log_likelihood
                paths = self.crf.decode(logits, mask=mask)
            else:
                # 增强模式：将 [CLS]（位置 0）排除在 CRF 之外，从索引 1 开始切片
                offset = 1
                logits_crf = logits[:, offset:, :]
                labels_crf = labels_pad[:, offset:]
                mask_crf = base_mask[:, offset:]
                if mask_crf.dim() == 2:
                    # 切片后仍需满足 torchcrf 约束：首个时间步有效
                    no_valid = mask_crf.sum(dim=1) == 0
                    # 若整条序列无有效位，将第一个位置置为 True
                    if no_valid.any():
                        mask_crf[no_valid, 0] = True
                    # 保证时间步 0 为 True
                    mask_crf[:, 0] = True
                log_likelihood = self.crf(logits_crf, labels_crf, mask=mask_crf, reduction="mean")
                loss = 0 - log_likelihood
                paths = self.crf.decode(logits_crf, mask=mask_crf)
                mask = mask_crf  # 用于结合 offset 回填 pred_tags
        else:
            if self.paper_aligned:
                if attention_mask is not None:
                    mask = attention_mask.bool()
                # 解码阶段保持自然的注意力掩码；不强制首位为 True
                paths = self.crf.decode(logits, mask=mask)
            else:
                # 增强模式：解码时同样排除 [CLS]
                offset = 1
                if attention_mask is not None:
                    base_mask = attention_mask.bool()
                else:
                    # 若未提供 attention_mask，默认所有 token 有效
                    base_mask = torch.ones(logits.size()[:2], dtype=torch.bool, device=logits.device)
                logits_crf = logits[:, offset:, :]
                mask_crf = base_mask[:, offset:]
                if mask_crf.dim() == 2:
                    # 满足 torchcrf 的首位有效约束
                    mask_crf[:, 0] = True
                paths = self.crf.decode(logits_crf, mask=mask_crf)
                mask = mask_crf

        # 将解码得到的路径回填为固定形状 (batch, seq_len) 的 LongTensor
        batch_size, seq_len = logits.shape[:2]
        pred_tags = torch.zeros((batch_size, seq_len), dtype=torch.long, device=logits.device)
        for i, path in enumerate(paths):
            if mask is not None:
                true_pos = mask[i].nonzero(as_tuple=False).view(-1)
                if true_pos.numel() > 0:
                    true_pos = true_pos + offset
                L = min(len(path), true_pos.numel())
                if L > 0:
                    pred_tags[i, true_pos[:L]] = torch.tensor(path[:L], dtype=torch.long, device=logits.device)
            else:
                # 无掩码：从 offset 开始顺序回填
                start = offset
                end = min(seq_len, offset + len(path))
                if end > start:
                    pred_tags[i, start:end] = torch.tensor(path[: end - start], dtype=torch.long, device=logits.device)

        # 保持返回元组形式，方便 Trainer 将解码标签传入 compute_metrics。
        # 第三个元素返回 logits 以便必要时检查。
        if not return_dict:
            return (loss, pred_tags, logits) if loss is not None else (pred_tags, logits)

        # 即便 return_dict=True，也依旧返回元组以保持当前 metrics 行为。
        # TokenClassifierOutput 已准备好，但 HF Trainer 在结合 CRF 的预测中并不会使用它。
        # 若需要返回 HF 风格输出，可取消注释（这会影响 compute_metrics 的入参约定）：
        # tco = TokenClassifierOutput(loss=loss, logits=logits,
        #                             hidden_states=getattr(outputs, "hidden_states", None),
        #                             attentions=getattr(outputs, "attentions", None))
        # return tco
        return (loss, pred_tags, logits) if loss is not None else (pred_tags, logits)

    # 保留与论文参考代码一致的函数名，便于对照阅读
    def init_weights(self):
        for name, param in self.bilstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

        nn.init.uniform_(self.crf.transitions, -0.1, 0.1)
        nn.init.uniform_(self.crf.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.crf.end_transitions, -0.1, 0.1)
