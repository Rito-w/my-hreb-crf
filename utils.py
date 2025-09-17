from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import TrainerCallback
import torch
import numpy as np


def tokenize(batch, tokenizer, label_names, max_length=64):
    tokenized_inputs = tokenizer(
        batch["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
    )

    # 根据标签字符串构建稳健的 B->I 映射，避免依赖奇偶 ID 的脆弱假设
    name_to_id = {name: idx for idx, name in enumerate(label_names)}
    to_inside = list(range(len(label_names)))
    for idx, name in enumerate(label_names):
        if isinstance(name, str) and name.startswith("B-"):
            i_name = "I-" + name[2:]
            to_inside[idx] = name_to_id.get(i_name, idx)
        else:
            to_inside[idx] = idx

    all_labels = []
    for i, labels_for_one_example in enumerate(batch["label_ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # 首个子词沿用原始词级标签 ID
                wid = labels_for_one_example[word_idx]
                wid = int(wid) if isinstance(wid, (int, np.integer)) else wid
                label_ids.append(wid)
            else:
                # 后续子词使用对应 I- 标签（若存在），否则回退为原 ID
                wid = labels_for_one_example[word_idx]
                wid = int(wid) if isinstance(wid, (int, np.integer)) else wid
                wid = wid if (0 <= wid < len(to_inside)) else 0
                label_ids.append(to_inside[wid])
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["label_ids"] = all_labels
    return tokenized_inputs


def compute_metrics(pred, dataset):
    labels = pred.label_ids.flatten()
    preds_arr = pred.predictions[0] if isinstance(pred.predictions, (list, tuple)) else pred.predictions
    preds = preds_arr.flatten().astype(int)
    _label = dataset["train"].features["ner_tags"].feature.names
    # 过滤掉被忽略的位点与纯 O-O 情况
    true_predictions = [
        _label[pred] for pred, label in zip(preds, labels) if label != -100 and not (label == 0 and pred == 0)
    ]

    true_labels = [
        _label[label] for pred, label in zip(preds, labels) if label != -100 and not (label == 0 and pred == 0)
    ]

    preds = true_predictions
    labels = true_labels
    filtered_labels = [l for l in _label if l != 'O']
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    p = precision_score(labels, preds, average='weighted', zero_division=0)
    r = recall_score(labels, preds, average='weighted', zero_division=0)
    print(classification_report(labels, preds, labels=filtered_labels, zero_division=0))

    return {
        'precision': p,
        'recall': r,
        'f1': f1,
    }


# 将 BIO 序列转为实体跨度列表（type, start, end）
def _bio_spans(seq):
    spans = []  # (type, start, end)
    cur_t, cur_s = None, None
    for i, tag in enumerate(seq):
        if tag == 'O' or tag is None:
            if cur_t is not None:
                spans.append((cur_t, cur_s, i))
                cur_t, cur_s = None, None
            continue
        if tag.startswith('B-'):
            if cur_t is not None:
                spans.append((cur_t, cur_s, i))
            cur_t, cur_s = tag[2:], i
        elif tag.startswith('I-'):
            t = tag[2:]
            if cur_t is None:
                cur_t, cur_s = t, i  # 容错：孤立的 I 当作新段开始
            elif t != cur_t:
                spans.append((cur_t, cur_s, i))
                cur_t, cur_s = t, i
        else:
            # 未知标签，直接断开当前段
            if cur_t is not None:
                spans.append((cur_t, cur_s, i))
                cur_t, cur_s = None, None
    if cur_t is not None:
        spans.append((cur_t, cur_s, len(seq)))
    return spans


def compute_metrics_entity(pred, dataset):
    names = dataset["train"].features["ner_tags"].feature.names
    preds_arr = pred.predictions[0] if isinstance(pred.predictions, (list, tuple)) else pred.predictions
    preds = preds_arr
    labels = pred.label_ids

    # 基于 -100 掩码，按句重建标签序列
    y_true, y_pred = [], []
    try:
        B, T = labels.shape
    except Exception:
        # 兜底：无法按句还原时，退回到旧的扁平化计算
        labels = labels.flatten()
        preds = preds.flatten().astype(int)
        gold_tags = [names[l] for l in labels if l != -100]
        pred_tags = [names[p] for p, l in zip(preds, labels) if l != -100]
        gold_spans = set(_bio_spans(gold_tags))
        pred_spans = set(_bio_spans(pred_tags))
        inter = len(gold_spans & pred_spans)
        p = inter / len(pred_spans) if len(pred_spans) else 0.0
        r = inter / len(gold_spans) if len(gold_spans) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"Entity-level (span) metrics -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
        return {"precision": p, "recall": r, "f1": f1}

    for i in range(B):
        true_row, pred_row = [], []
        for l, p in zip(labels[i].tolist(), preds[i].tolist()):
            if l == -100:
                continue
            true_row.append(names[int(l)])
            pred_row.append(names[int(p)])
        if true_row:
            y_true.append(true_row)
            y_pred.append(pred_row)

    # 优先使用 seqeval（标准实体级指标）；不可用时回退到 span 级实现
    try:
        from seqeval.metrics import precision_score as seqeval_precision
        from seqeval.metrics import recall_score as seqeval_recall
        from seqeval.metrics import f1_score as seqeval_f1
        p = float(seqeval_precision(y_true, y_pred))
        r = float(seqeval_recall(y_true, y_pred))
        f1 = float(seqeval_f1(y_true, y_pred))
        print(f"SeqEval entity-level -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
        return {"precision": p, "recall": r, "f1": f1}
    except Exception:
        # span 级回退（将每个句子视为独立序列）
        gold_spans_all, pred_spans_all = set(), set()
        for gt, pd in zip(y_true, y_pred):
            gold_spans_all |= set(_bio_spans(gt))
            pred_spans_all |= set(_bio_spans(pd))
        inter = len(gold_spans_all & pred_spans_all)
        p = inter / len(pred_spans_all) if len(pred_spans_all) else 0.0
        r = inter / len(gold_spans_all) if len(gold_spans_all) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"Entity-level (span) metrics -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
        return {"precision": p, "recall": r, "f1": f1}


class WeightLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            model = kwargs.get('model')
            if model is not None:
                # 打印当前门控权重（sigmoid 后的值），便于训练过程观察
                r_lstm = torch.sigmoid(model.r_lstm).item()
                r_mega = torch.sigmoid(model.r_mega).item()
                print(f"Current weight of:")
                print(f"r_lstm: {r_lstm:.4f}")
                print(f"r_mega: {r_mega:.4f}\n")
