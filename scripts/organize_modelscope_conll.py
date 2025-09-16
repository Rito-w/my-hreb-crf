import os
from pathlib import Path

ROOT = Path(os.getcwd())
RAW = ROOT / "data" / "raw" / "modelscope"
OUT = ROOT / "data" / "corpora"


def normalize_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", errors="ignore") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            s = line.rstrip("\n\r")
            if not s:
                f_out.write("\n")
                continue
            s = s.replace("\t", " ")
            f_out.write(s + "\n")


def handle_simple(repo: str, out_name: str):
    src_dir = RAW / repo
    if not src_dir.exists():
        print(f"[WARN] missing {src_dir}")
        return
    mapping = {
        "train.txt": OUT / out_name / "train.conll",
        "dev.txt": OUT / out_name / "dev.conll",
        "test.txt": OUT / out_name / "test.conll",
    }
    for src_name, dst_path in mapping.items():
        src = src_dir / src_name
        if src.exists():
            normalize_copy(src, dst_path)
            print(f"Wrote {dst_path} from {src}")
        else:
            print(f"[WARN] not found: {src}")


def handle_cross():
    base = RAW / "cross_ner"
    if not base.exists():
        print(f"[WARN] missing {base}")
        return
    for domain in ["ai", "literature", "music", "politics", "science"]:
        ddir = base / domain
        if not ddir.exists():
            print(f"[WARN] skip domain {domain}, path missing: {ddir}")
            continue
        out_dir = OUT / "cross_ner" / domain
        for split in ["train", "dev", "test"]:
            src = ddir / f"{split}.txt"
            dst = out_dir / f"{split}.conll"
            if src.exists():
                normalize_copy(src, dst)
                print(f"Wrote {dst} from {src}")
            else:
                print(f"[WARN] not found: {src}")


def main():
    handle_simple("weibo_ner", "weibo")
    handle_simple("resume_ner", "resume")
    handle_cross()
    print("Done.")


if __name__ == "__main__":
    main()

