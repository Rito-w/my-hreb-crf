# my-hreb-crf

本仓库在保持参考实现 `HREB-CRF/` 相似高层结构（BERT/RoBERTa 骨干 + MEGA(EMA) + BiLSTM + CRF）的基础上，进行了更实用、更加鲁棒的工程化改造。主要改进包括：更稳健的掩码与标签对齐、可复现性、可选的动态 padding；同时提供“论文对齐”开关，便于在论文行为与增强实现之间切换。

> 说明：论文中的完整 HEMA（Hierarchical Reduced-bias EMA）与“降低偏置”残差尚未完全实现；本项目旨在提供一个可靠的基线与工程友好的起点，后续会逐步补齐 HEMA/Reduced-bias 模块。

## 快速开始（Quickstart）

1) 创建环境并安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Windows：.venv\Scripts\activate
pip install -r my-hreb-crf/requirements.txt
```

2) 准备数据（可选：若你已有本地 HF 数据集可跳过）

- 将 CoNLL 文件放置到 `data/corpora/<dataset>/{train,dev,test}.conll`
- 转换为本地 HuggingFace Datasets，保存到 `data/hf/local/<dataset>`：

```bash
python my-hreb-crf/scripts/conll_to_hf.py \
  --src data/corpora/msra \
  --dst data/hf/local/msra
```

3) 运行单次实验

```bash
# 推荐使用 RoBERTa 骨干
export BERT_MODEL=hfl/chinese-roberta-wwm-ext-large

# 可选开关
export MODEL_PAPER_ALIGNED=0   # 1 更接近参考实现；0 为增强模式
export DYNAMIC_PADDING=1       # 动态 padding 提升效率（默认 1）
export METRIC=entity           # 也可设为 token

python my-hreb-crf/run_eval.py
```

4) 批量评测本地多个数据集

```bash
python my-hreb-crf/scripts/batch_eval_all.py
```

结果将保存到 `my-hreb-crf/results/{model}-{dataset}-{timestamp}`，同时包含实体级（若安装 seqeval 则优先使用）与 token 级指标。

## 环境变量（Environment variables）

- MODEL_PAPER_ALIGNED：`1/true/on` 开启论文对齐；`0/false/off` 使用增强模式。默认：None（等价论文对齐）。
- DYNAMIC_PADDING：`1/0` 通过 `DataCollatorForTokenClassification` 启用/关闭动态 padding。默认：`1`。
- MEGA_LAPLACIAN：`1/0` 启用/关闭 `MegaLayer` 的拉普拉斯注意力，用于更贴近论文的试验。默认：`0`。
- REDUCED_BIAS：`1/0` 启用/关闭动态残差门控（降低偏置残差原型）。默认：`0`。
- BERT_MODEL：HF 模型 ID，例如 `hfl/chinese-roberta-wwm-ext-large` 或 `bert-base-chinese`。
- DATASET_PATH：优先指定本地 HF 数据集路径（如 `data/hf/local/msra`）。若未设置，脚本会尝试 `data/hf/local/{msra,resume,weibo,cross_ner_all}`，再退回至远端或离线迷你集。
- TRAIN_BS/EVAL_BS/GRAD_ACC/NUM_EPOCHS/MAX_LENGTH/LEARNING_RATE：常规训练超参数。
- METRIC：`entity`（默认）或 `token`。
- EVAL_SAVE：`1` 保存检查点；默认 `0` 以节省磁盘。
- SEED：随机种子（默认 42）。
- TRANSFORMERS_OFFLINE/HF_OFFLINE：设为 `1` 以启用离线模式。

## 模式（Modes）

- 论文对齐模式（reference-like）：
  - 不向 MEGA 传入 attention_mask；r_lstm/r_mega 不做 sigmoid 约束。
  - CRF 训练遵循 torchcrf 对首位有效的要求；解码阶段保持自然掩码。

- 增强模式（推荐）：
  - 在支持的情况下，为 MEGA 传入布尔 attention_mask。
  - r_lstm/r_mega 经过 sigmoid 保持在 [0,1]。
  - 解码阶段不强制首位有效。
  - 可选：设置 `REDUCED_BIAS=1`，在骨干输出与融合分支之间启用动态残差门控。

## 数据与脚本（Data & scripts）
- 将 CoNLL 语料放到 `./data/corpora/<dataset>/{train,dev,test}.conll`
- 运行脚本位于 `./scripts`（batch_eval_all.py、转换脚本等）
- 体积较大的 HF 数据集不纳入版本控制；可通过 `scripts/conll_to_hf.py` 重新生成。

## 说明（Notes）

- 官方参考代码位于 `HREB-CRF/`。本仓库（`my-hreb-crf/`）是工程化改进版；尚未完全实现论文中的 HEMA/降低偏置模块。
- 实体级指标若安装了 `seqeval` 则优先采用；否则回退为 span 级实现。
- `MEGA_LAPLACIAN` 仅打开 `MegaLayer` 的拉普拉斯注意力开关；论文中的“降低偏置”拉普拉斯公式尚未完全实现。
