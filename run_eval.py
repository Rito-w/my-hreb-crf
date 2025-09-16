import os
import time
import json

import torch
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from transformers import BertTokenizerFast, Trainer, TrainingArguments

# Import model and utils from the current directory (new-herb-crf)
from model import HREBCRF
from utils import tokenize, compute_metrics, compute_metrics_entity, WeightLoggerCallback


def try_load_dataset(name: str):
    print(f"Loading dataset: {name}")
    # If 'name' points to a local saved dataset directory, prefer local
    if os.path.isdir(name):
        return load_from_disk(name)
    return load_dataset(name)



def pick_local_dataset() -> str | None:
    """Prefer a local HF dataset under data/hf/local.
    Order of precedence:
    1) DATASET_LOCAL env (e.g., msra/resume/weibo/cross_ner_all)
    2) First existing among [msra, resume, weibo, cross_ner_all]
    Returns path or None if not found.
    """
    root = os.path.join("data", "hf", "local")
    prefer = os.environ.get("DATASET_LOCAL")
    candidates = []
    if prefer:
        candidates.append(os.path.join(root, prefer))
    candidates += [os.path.join(root, name) for name in ["msra", "resume", "weibo", "cross_ner_all"]]
    for p in candidates:
        if os.path.isdir(p):
            print(f"Auto-selected local dataset: {p}")
            return p
    return None


def build_tiny_ner_dataset() -> DatasetDict:
    print("Building tiny offline NER dataset (zh)...")
    label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_list)),
    })

    train_tokens = [
        list("张三在北京工作"),
        list("李四来到了清华大学"),
        list("他在阿里巴巴实习"),
        list("王五去上海旅游"),
        list("张三在阿里巴巴工作"),
        list("李四在北京上学"),
        list("他去了清华大学"),
        list("王五在上海实习"),
    ]
    train_tags = [
        [1, 2, 0, 3, 4, 0, 0],
        [1, 2, 0, 0, 0, 5, 6, 6, 6],
        [0, 0, 5, 6, 6, 6, 0, 0],
        [1, 2, 0, 3, 4, 0, 0],
        [1, 2, 0, 5, 6, 6, 6, 0, 0],
        [1, 2, 0, 3, 4, 0, 0],
        [0, 0, 0, 5, 6, 6, 6],
        [1, 2, 0, 3, 4, 0, 0],
    ]

    test_tokens = [
        list("赵六在北京上班"),
        list("他在华为工作"),
        list("钱七去了上海"),
        list("李四在阿里工作"),
    ]
    test_tags = [
        [1, 2, 0, 3, 4, 0, 0],
        [0, 0, 5, 6, 0, 0],
        [1, 2, 0, 3, 4],
        [1, 2, 0, 5, 6, 0, 0],
    ]

    train_ds = Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_tags}, features=features)
    test_ds = Dataset.from_dict({"tokens": test_tokens, "ner_tags": test_tags}, features=features)

    return DatasetDict({"train": train_ds, "test": test_ds})


def main():
    # Configs
    dataset_name = os.environ.get("DATASET_NAME", "PassbyGrocer/msra-ner")
    fallback_name = os.environ.get("FALLBACK_DATASET", "wikiann")
    fallback_config = os.environ.get("FALLBACK_CONFIG", "zh")
    bert_model = os.environ.get("BERT_MODEL", "bert-base-chinese")
    max_length = int(os.environ.get("MAX_LENGTH", 64))
    train_samples = int(os.environ.get("TRAIN_SAMPLES", 400))
    eval_samples = int(os.environ.get("EVAL_SAMPLES", 400))
    num_epochs = float(os.environ.get("NUM_EPOCHS", 1))
    per_device_train_bs = int(os.environ.get("TRAIN_BS", 8))
    per_device_eval_bs = int(os.environ.get("EVAL_BS", 8))
    grad_acc = int(os.environ.get("GRAD_ACC", 1))
    metric_kind = os.environ.get("METRIC", "token")  # 'token' or 'entity'

    # Load dataset with multi-stage fallback (prefer local under data/hf/local)
    def _safe_name(s: str) -> str:
        s = s.replace(os.sep, "-").replace(" ", "_")
        return "".join(c if c.isalnum() or c in ("-", "_", ".") else "-" for c in s)

    dataset_path = os.environ.get("DATASET_PATH") or pick_local_dataset()
    dataset_key = None
    try:
        if dataset_path and os.path.isdir(dataset_path):
            print(f"Loading local dataset from: {dataset_path}")
            dataset = load_from_disk(dataset_path)
            dataset_key = os.path.basename(dataset_path.rstrip("/"))
        else:
            dataset = try_load_dataset(dataset_name)
            dataset_key = dataset_name
    except Exception as e:
        print(f"Primary dataset load failed: {e}\nFalling back to '{fallback_name}' ('{fallback_config}')...")
        try:
            dataset = load_dataset(fallback_name, fallback_config)
            dataset_key = f"{fallback_name}-{fallback_config}"
        except Exception as e2:
            print(f"Fallback '{fallback_name}' failed: {e2}\nUsing tiny offline dataset instead.")
            dataset = build_tiny_ner_dataset()
            dataset_key = "tiny-offline"

    # Decide train/eval splits
    train_dataset = dataset["train"]
    test_dataset = dataset["test"] if "test" in dataset else dataset.get("validation", dataset["train"])  # fallback

    # Subset for speed (skip for tiny dataset)
    if len(train_dataset) > 100:
        if train_samples > 0 and train_samples < len(train_dataset):
            train_dataset = train_dataset.select(range(train_samples))
    if len(test_dataset) > 100:
        if eval_samples > 0 and eval_samples < len(test_dataset):
            test_dataset = test_dataset.select(range(eval_samples))

    label_names = train_dataset.features["ner_tags"].feature.names
    num_labels = len(label_names)
    print(f"Num labels: {num_labels}")

    print(f"Loading tokenizer and model: {bert_model}")
    offline = os.environ.get("TRANSFORMERS_OFFLINE", os.environ.get("HF_OFFLINE", "0")) == "1"
    tokenizer = BertTokenizerFast.from_pretrained(bert_model, local_files_only=offline)
    model = HREBCRF.from_pretrained(bert_model, num_labels=num_labels, local_files_only=offline)

    # Resolve run-specific output directory: results/{model}-{dataset}-{timestamp}
    model_key = os.path.basename(bert_model.rstrip("/")) if os.path.sep in bert_model else bert_model
    model_key = _safe_name(model_key)
    ds_key_env = os.environ.get("DATASET_LOCAL")
    if ds_key_env and dataset_key is None:
        dataset_key = ds_key_env
    dataset_key = _safe_name(dataset_key or "dataset")
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_root = os.environ.get("OUTPUT_DIR_ROOT", "./new-herb-crf/results")
    out_dir = os.path.join(out_root, f"{model_key}-{dataset_key}-{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Prepare tokenized datasets
    # 检查标签类型并转换
    def convert_ner_tags_to_ids(dataset):
        first_example = dataset[0]
        first_label = first_example["ner_tags"][0]
        if isinstance(first_label, str):
            print("Detected string labels, converting to numeric IDs...")
            label2id = {label: i for i, label in enumerate(dataset.features["ner_tags"].feature.names)}
            print(f"Label mapping: {label2id}")
            def convert_example(example):
                example["label_ids"] = [label2id[label] for label in example["ner_tags"]]
                return example
            return dataset.map(convert_example)
        else:
            print("Detected numeric labels, copying to label_ids...")
            def copy_example(example):
                example["label_ids"] = example["ner_tags"].copy()
                return example
            return dataset.map(copy_example)

    print("Processing labels...")
    train_dataset = convert_ner_tags_to_ids(train_dataset)
    test_dataset = convert_ner_tags_to_ids(test_dataset)

    # 删除原始的ner_tags列，因为已经创建了label_ids
    train_dataset = train_dataset.remove_columns(["ner_tags"])
    test_dataset = test_dataset.remove_columns(["ner_tags"])

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize(x, tokenizer, label_names, max_length),
        batched=True,
        batch_size=min(1024, len(train_dataset)) or 1,
        desc="Tokenizing train",
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize(x, tokenizer, label_names, max_length),
        batched=True,
        batch_size=min(1024, len(test_dataset)) or 1,
        desc="Tokenizing eval",
    )

    cols = ["input_ids", "token_type_ids", "attention_mask", "label_ids"]
    train_dataset.set_format("torch", columns=cols)
    test_dataset.set_format("torch", columns=cols)

    use_gpu = torch.cuda.is_available()
    print(f"CUDA available: {use_gpu}")

    # Allow toggling fp16 via env (default off)
    use_fp16 = os.environ.get("EVAL_FP16", "0") == "1"

    # allow disabling checkpoint/save to avoid disk pressure
    save_enabled = os.environ.get("EVAL_SAVE", "0") == "1"
    save_strategy = "epoch" if save_enabled else "no"

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_bs,
        per_device_eval_batch_size=per_device_eval_bs,
        gradient_accumulation_steps=grad_acc,
        learning_rate=float(os.environ.get("LEARNING_RATE", 3e-5)),
        evaluation_strategy="epoch",
        logging_steps=10,
        save_strategy=save_strategy,
        save_total_limit=2 if save_enabled else 0,
        load_best_model_at_end=save_enabled,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
        fp16=use_gpu and use_fp16,
    )

    metric_fn = (lambda x: compute_metrics_entity(x, dataset)) if metric_kind == "entity" else (lambda x: compute_metrics(x, dataset))
    trainer = Trainer(
        model=model,
        args=args,
        compute_metrics=metric_fn,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[WeightLoggerCallback()],
    )

    start = time.time()
    print("Starting training...")
    trainer.train(resume_from_checkpoint=False)
    print("Training finished. Running evaluation...")
    metrics = trainer.evaluate()
    elapsed = time.time() - start

    # Persist metrics (and optionally model/tokenizer if saving is enabled)
    os.makedirs(out_dir, exist_ok=True)
    try:
        trainer.save_metrics("eval", metrics)
    except Exception as e:
        print(f"Warning: save_metrics failed: {e}")

    if os.environ.get("EVAL_SAVE", "0") == "1":
        try:
            trainer.save_state()
            trainer.save_model(out_dir)
            tokenizer.save_pretrained(out_dir)
        except Exception as e:
            print(f"Warning: save_model/save_state failed: {e}")

    predictions = trainer.predict(test_dataset)
    # 两份指标：实体级 与 token 级
    metrics_entity = compute_metrics_entity(predictions, dataset)
    metrics_token = compute_metrics(predictions, dataset)
    with open(os.path.join(out_dir, "metrics_entity.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_entity, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "metrics_token.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_token, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to: {out_dir}")

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

