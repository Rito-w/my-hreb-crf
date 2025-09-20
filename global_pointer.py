import math
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F


def _build_entity_types(id2label: Dict[int, str]) -> Tuple[List[str], Dict[int, int], Dict[str, Tuple[int, int]]]:
    """From BIO label space build entity type list and mappings.
    Returns:
      - types: list of unique entity types (e.g., ["PER","LOC",...])
      - label2type: map from label id -> type index (or -1 for 'O')
      - type2bio_ids: type -> (B_id, I_id)
    """
    types: List[str] = []
    type2bio_ids: Dict[str, Tuple[int, int]] = {}
    # Collect types from B-/I-
    for i, name in id2label.items():
        if isinstance(name, str) and len(name) > 2 and name[1] == '-' and name[0] in ('B', 'I'):
            t = name[2:]
            if t not in type2bio_ids:
                type2bio_ids[t] = (-1, -1)
            if name.startswith('B-'):
                type2bio_ids[t] = (i, type2bio_ids[t][1])
            elif name.startswith('I-'):
                type2bio_ids[t] = (type2bio_ids[t][0], i)
    types = sorted(type2bio_ids.keys())
    type_index = {t: idx for idx, t in enumerate(types)}
    label2type: Dict[int, int] = {}
    for i, name in id2label.items():
        if name == 'O' or not isinstance(name, str):
            label2type[i] = -1
        elif name.startswith('B-') or name.startswith('I-'):
            t = name[2:]
            label2type[i] = type_index.get(t, -1)
        else:
            label2type[i] = -1
    return types, label2type, type2bio_ids


def build_gp_label_mappings(id2label: Dict[int, str]):
    return _build_entity_types(id2label)


def make_gp_targets(labels: torch.Tensor, label2type: Dict[int, int], token_mask: torch.Tensor, num_types: int) -> torch.Tensor:
    """Build target tensor of shape [B, C, L, L] from BIO labels
    labels: [B, L] with -100 for ignored
    token_mask: [B, L] bool for valid tokens (ignored positions False)
    """
    B, L = labels.shape
    tgt = torch.zeros((B, num_types, L, L), dtype=torch.float32, device=labels.device)
    labels_cpu = labels.detach().cpu().tolist()
    mask_cpu = token_mask.detach().cpu().tolist()
    for b in range(B):
        seq = labels_cpu[b]
        msk = mask_cpu[b]
        i = 0
        while i < L:
            if not msk[i]:
                i += 1
                continue
            lab = seq[i]
            t_idx = label2type.get(lab, -1)
            if t_idx >= 0:
                # start of a span if this is B- or standalone I- (tolerant)
                # We treat both B- and I- as potential start for robustness
                j = i
                while j + 1 < L and msk[j + 1]:
                    next_lab = seq[j + 1]
                    if label2type.get(next_lab, -2) != t_idx:
                        break
                    j += 1
                tgt[b, t_idx, i, j] = 1.0
                i = j + 1
            else:
                i += 1
    return tgt


def gp_decode_to_bio(
    logits: torch.Tensor,
    token_mask: torch.Tensor,
    type2bio_ids: Dict[str, Tuple[int, int]],
    types: List[str],
    threshold: float = 0.0,
) -> torch.Tensor:
    """Greedy non-overlapping decode to BIO tag ids.
    logits: [B, C, L, L]
    token_mask: [B, L] bool
    Returns: [B, L] Long (BIO tag ids)
    """
    B, C, L, _ = logits.shape
    device = logits.device
    pred_tags = torch.zeros((B, L), dtype=torch.long, device=device)  # default 'O' assumed to be id 0

    # Build helper maps from type index to (B_id, I_id)
    t2bio = []
    for t in types:
        b_id, i_id = type2bio_ids.get(t, (-1, -1))
        t2bio.append((b_id, i_id))

    with torch.no_grad():
        for b in range(B):
            mask_b = token_mask[b]
            spans = []  # (score, s, e, t_idx)
            # Collect candidate spans above threshold and valid (s<=e)
            logit_b = logits[b]
            for t_idx in range(C):
                score_mat = logit_b[t_idx]
                # upper triangle including diagonal
                scores = score_mat.triu(diagonal=0)
                # Apply threshold
                cand = (scores > threshold).nonzero(as_tuple=False)
                for idx in cand:
                    s, e = int(idx[0]), int(idx[1])
                    if s >= L or e >= L:
                        continue
                    if mask_b[s].item() and mask_b[e].item():
                        spans.append((float(scores[s, e].item()), s, e, t_idx))
            # Greedy select non-overlapping spans by score
            spans.sort(key=lambda x: -x[0])
            used = torch.zeros(L, dtype=torch.bool, device=device)
            for _, s, e, t_idx in spans:
                if s > e:
                    continue
                if not mask_b[s].item():
                    continue
                if not mask_b[e].item():
                    continue
                if used[s:e + 1].any():
                    continue
                b_id, i_id = t2bio[t_idx]
                if b_id < 0 or i_id < 0:
                    continue
                pred_tags[b, s] = b_id
                if e > s:
                    pred_tags[b, s + 1:e + 1] = i_id
                used[s:e + 1] = True
            # Keep ignored positions (mask False) as 0; caller can map -100 if needed
    return pred_tags


class RoPEPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Precompute inverse frequency
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim).float() / half_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., L, D]
        L = x.size(-2)
        t = torch.arange(L, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('l,d->ld', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [L, D]
        cos = emb.cos()[None, ...]  # [1, L, D]
        sin = emb.sin()[None, ...]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack([x1 * cos[..., :x1.size(-1)] - x2 * sin[..., :x1.size(-1)],
                             x1 * sin[..., :x1.size(-1)] + x2 * cos[..., :x1.size(-1)]], dim=-1)
        return x_rot.flatten(-2)


class GlobalPointer(nn.Module):
    """GlobalPointer head (with optional RoPE) producing span logits [B, C, L, L].
    Based on the original implementation idea by Su Jianlin / Gao Hongkui.
    """
    def __init__(self, hidden_size: int, heads: int, head_size: int = 64, rope: bool = True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.rope = rope
        self.dense = nn.Linear(hidden_size, heads * head_size * 2)
        self.rope_emb = RoPEPositionEmbedding(head_size) if rope else None

    def forward(self, h: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: [B, L, H]
        B, L, H = h.size()
        out = self.dense(h)  # [B, L, C*2*Hd]
        out = out.view(B, L, self.heads, 2, self.head_size)
        qw, kw = out[..., 0, :], out[..., 1, :]  # [B, L, C, Hd]
        # Move heads before length for einsum convenience
        qw = qw.permute(0, 2, 1, 3)  # [B, C, L, Hd]
        kw = kw.permute(0, 2, 1, 3)  # [B, C, L, Hd]
        if self.rope:
            qw = self.rope_emb(qw)
            kw = self.rope_emb(kw)
        # Bilinear scores for all span pairs
        logits = torch.einsum('bclh,bcnh->bcln', qw, kw)  # [B, C, L, L]
        logits = logits / math.sqrt(self.head_size)
        # Mask invalid tokens
        if token_mask is not None:
            m = token_mask.bool()
            # Only keep pairs on valid tokens
            # [B, 1, L, 1] and [B, 1, 1, L]
            m1 = m[:, None, :, None]
            m2 = m[:, None, None, :]
            pair_mask = m1 & m2
            logits = logits.masked_fill(~pair_mask, -1e12)
        # Keep only upper triangle (end >= start)
        triu_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device))
        logits = logits.masked_fill(~triu_mask[None, None, :, :], -1e12)
        return logits


def gp_loss_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    # Ignore positions with -1e12 logits (masked out), which will have target=0 and huge negative logit
    # Clip to avoid NaNs
    loss = torch.where(logits < -1e11, torch.zeros_like(loss), loss)
    denom = (targets > 0).float().sum().clamp_min(1.0)
    return loss.sum() / denom

