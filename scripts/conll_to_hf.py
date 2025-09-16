import os
from pathlib import Path
from typing import List, Tuple, Dict

from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value

ROOT = Path(os.getcwd())
CORPORA = ROOT / "data" / "corpora"
HF_OUT_ROOT = ROOT / "data" / "hf" / "local"
HF_OUT_ROOT.mkdir(parents=True, exist_ok=True)


def parse_conll(path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    tokens_batch: List[List[str]] = []
    tags_batch: List[List[str]] = []
    cur_toks: List[str] = []
    cur_tags: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                if cur_toks:
                    tokens_batch.append(cur_toks)
                    tags_batch.append(cur_tags)
                    cur_toks, cur_tags = [], []
                continue
            parts = line.split()
            if len(parts) == 1:
                tok, tag = parts[0], "O"
            else:
                tok, tag = parts[0], parts[-1]
            cur_toks.append(tok)
            cur_tags.append(tag)
    if cur_toks:
        tokens_batch.append(cur_toks)
        tags_batch.append(cur_tags)
    return tokens_batch, tags_batch


def collect_labels(*splits_paths: List[Path]) -> List[str]:
    labels = set(["O"])  # ensure O exists
    for p in splits_paths:
        if p and p.exists():
            _, tags_batch = parse_conll(p)
            for tags in tags_batch:
                for t in tags:
                    labels.add(t)
    # put O first, then sorted others for stability
    rest = sorted([x for x in labels if x != "O"])
    return ["O"] + rest


def build_dataset(train_p: Path, dev_p: Path, test_p: Path) -> DatasetDict:
    labels = collect_labels(train_p, dev_p, test_p)
    feats = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=labels)),
    })

    def to_ds(p: Path) -> Dataset:
        if not p.exists():
            return Dataset.from_dict({"tokens": [], "ner_tags": []}, features=feats)
        toks, tags = parse_conll(p)
        # map tags to ids via ClassLabel
        name2id = {n: i for i, n in enumerate(labels)}
        tags_ids = [[name2id[t] for t in seq] for seq in tags]
        return Dataset.from_dict({"tokens": toks, "ner_tags": tags_ids}, features=feats)

    ds = DatasetDict()
    if train_p is not None:
        ds["train"] = to_ds(train_p)
    if dev_p is not None:
        ds["validation"] = to_ds(dev_p)
    if test_p is not None:
        ds["test"] = to_ds(test_p)
    return ds


def export_one(name: str, base_dir: Path, out_dir: Path):
    train_p = base_dir / "train.conll"
    dev_p = base_dir / "dev.conll"
    test_p = base_dir / "test.conll"
    ds = build_dataset(train_p, dev_p, test_p)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"Saved HF dataset: {name} -> {out_dir}")
    for split, d in ds.items():
        print(f"  {split}: {len(d)}")


def main():
    # msra, weibo, resume
    for name in ["msra", "weibo", "resume"]:
        base = CORPORA / name
        if base.exists():
            export_one(name, base, HF_OUT_ROOT / name)
        else:
            print(f"[WARN] missing corpus: {base}")

    # cross_ner domains
    cross_base = CORPORA / "cross_ner"
    if cross_base.exists():
        domains = [d.name for d in cross_base.iterdir() if d.is_dir()]
        for dom in sorted(domains):
            base = cross_base / dom
            export_one(f"cross_ner/{dom}", base, HF_OUT_ROOT / "cross_ner" / dom)
        # optional: merged cross_ner_all
        try:
            # merge by concatenation
            import itertools
            from datasets import concatenate_datasets
            # ensure same label space by rebuilding with union labels
            # Collect union labels across domains
            all_paths = []
            for dom in domains:
                p = cross_base / dom
                all_paths += [p / "train.conll", p / "dev.conll", p / "test.conll"]
            union_labels = collect_labels(*all_paths)
            feats_all = Features({
                "tokens": Sequence(Value("string")),
                "ner_tags": Sequence(ClassLabel(names=union_labels)),
            })
            name2id_all = {n: i for i, n in enumerate(union_labels)}

            def to_ds(p: Path) -> Dataset:
                if not p.exists():
                    return Dataset.from_dict({"tokens": [], "ner_tags": []}, features=feats_all)
                toks, tags = parse_conll(p)
                tags_ids = [[name2id_all[t] for t in seq] for seq in tags]
                return Dataset.from_dict({"tokens": toks, "ner_tags": tags_ids}, features=feats_all)

            train_list = [to_ds((cross_base / d / "train.conll")) for d in domains]
            dev_list = [to_ds((cross_base / d / "dev.conll")) for d in domains]
            test_list = [to_ds((cross_base / d / "test.conll")) for d in domains]

            merged = DatasetDict({
                "train": concatenate_datasets(train_list),
                "validation": concatenate_datasets(dev_list),
                "test": concatenate_datasets(test_list),
            })
            out_all = HF_OUT_ROOT / "cross_ner_all"
            out_all.mkdir(parents=True, exist_ok=True)
            merged.save_to_disk(str(out_all))
            print(f"Saved HF dataset: cross_ner_all -> {out_all}")
            for split, d in merged.items():
                print(f"  {split}: {len(d)}")
        except Exception as e:
            print(f"[WARN] merge cross_ner_all failed: {e}")


if __name__ == "__main__":
    main()

