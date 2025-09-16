from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torch import nn
import torch
from torchcrf import CRF
from mega_pytorch import MegaLayer
import inspect


class HREBCRF(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
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
        self.mega = MegaLayer(
            dim=config.hidden_size,
            ema_heads=32,
            attn_dim_qk=128,
            attn_dim_value=256,
            laplacian_attn_fn=False,
        )
        # cache MegaLayer mask kwarg name once to avoid per-step try/except
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
        self.init_custom_weights()

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

        # last hidden
        bert_output = outputs[0]
        bert_output = self.dropout(bert_output)

        # LSTM branch
        lstm_output, _ = self.bilstm(bert_output)

        # Mega branch (pass cached mask kw if supported)
        if self._mega_mask_kw is not None and attention_mask is not None:
            mega_output = self.mega(bert_output, **{self._mega_mask_kw: attention_mask.bool()})
        else:
            mega_output = self.mega(bert_output)

        # gate with sigmoid to keep in [0,1]
        r_lstm = torch.sigmoid(self.r_lstm)
        r_mega = torch.sigmoid(self.r_mega)

        lstm_output = r_lstm * lstm_output + (1 - r_lstm) * bert_output
        mega_output = r_mega * mega_output + (1 - r_mega) * bert_output

        merge_output = self.layer_norm(mega_output + lstm_output)

        logits = self.classifier(merge_output)

        # CRF with proper mask and padding-safe labels
        loss = None
        mask = None
        paths = None
        if labels is not None:
            # make a copy with padding ids at 0; ensure long dtype
            labels_pad = labels.masked_fill(labels == -100, 0).to(dtype=torch.long)
            # build mask from labels/attention_mask; valid where label != -100 (and attention on if provided)
            if attention_mask is not None:
                mask = attention_mask.bool() & (labels != -100)
            else:
                mask = (labels != -100)
            # ensure the first timestep is ON for all sequences (torchcrf requirement)
            if mask.dim() == 2:
                # minimal fix for empty masks
                no_valid = mask.sum(dim=1) == 0
                if no_valid.any():
                    if attention_mask is not None:
                        first_idx = attention_mask.bool().float().argmax(dim=1)
                    else:
                        first_idx = torch.zeros(mask.size(0), dtype=torch.long, device=mask.device)
                    mask[torch.arange(mask.size(0), device=mask.device), first_idx] = True
                # torchcrf strict requirement: mask at time 0 must be True
                mask[:, 0] = True
            log_likelihood = self.crf(logits, labels_pad, mask=mask, reduction="mean")
            loss = 0 - log_likelihood
            paths = self.crf.decode(logits, mask=mask)
        else:
            if attention_mask is not None:
                mask = attention_mask.bool()
                if mask.dim() == 2:
                    mask[:, 0] = True
            paths = self.crf.decode(logits, mask=mask)

        # Pad decoded paths to fixed (batch, seq_len) LongTensor on correct device
        batch_size, seq_len = logits.shape[:2]
        pred_tags = torch.zeros((batch_size, seq_len), dtype=torch.long, device=logits.device)
        for i, path in enumerate(paths):
            if mask is not None:
                true_pos = mask[i].nonzero(as_tuple=False).view(-1)
                L = min(len(path), true_pos.numel())
                if L > 0:
                    pred_tags[i, true_pos[:L]] = torch.tensor(path[:L], dtype=torch.long, device=logits.device)
            else:
                L = min(len(path), seq_len)
                if L > 0:
                    pred_tags[i, :L] = torch.tensor(path[:L], dtype=torch.long, device=logits.device)

        # Keep tuple return to make Trainer feed decoded tags to compute_metrics.
        # Also include logits as the 3rd element for inspection.
        if not return_dict:
            return (loss, pred_tags, logits) if loss is not None else (pred_tags, logits)

        # If return_dict=True, we still return a tuple to preserve metrics behavior.
        # TokenClassifierOutput is prepared but not used by HF Trainer for predictions with CRF.
        # Uncomment below to return HF-style outputs (will change compute_metrics expectations):
        # tco = TokenClassifierOutput(loss=loss, logits=logits,
        #                             hidden_states=getattr(outputs, "hidden_states", None),
        #                             attentions=getattr(outputs, "attentions", None))
        # return tco
        return (loss, pred_tags, logits) if loss is not None else (pred_tags, logits)

    def init_custom_weights(self):
        # Only initialize the newly-added layers; avoid interfering with HF base init
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
