import os
from datasets import load_dataset

ROOT = os.getcwd()
HF_HOME = os.path.join(ROOT, "data", "hf")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUT_DIR = os.path.join(ROOT, "data", "corpora", "msra")
os.makedirs(OUT_DIR, exist_ok=True)


def write_conll(ds_split, path):
    names = ds_split.features["ner_tags"].feature.names
    with open(path, "w", encoding="utf-8") as f:
        for ex in ds_split:
            tokens = ex["tokens"]
            tags = [names[i] for i in ex["ner_tags"]]
            for t, y in zip(tokens, tags):
                f.write(f"{t} {y}\n")
            f.write("\n")


def main():
    ds = load_dataset("PassbyGrocer/msra-ner")
    write_conll(ds["train"], os.path.join(OUT_DIR, "train.conll"))
    split_val = ds.get("validation", ds["test"])  # use val if exists
    write_conll(split_val, os.path.join(OUT_DIR, "dev.conll"))
    write_conll(ds["test"], os.path.join(OUT_DIR, "test.conll"))
    print("Exported MSRA to:", OUT_DIR)


if __name__ == "__main__":
    main()

