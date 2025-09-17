import os
import json
import time
import subprocess
from pathlib import Path

ROOT = Path(os.getcwd())
BASE = ROOT / "my-hreb-crf"
PY = "python"
RUN_SCRIPT = ROOT / "my-hreb-crf" / "run_eval.py"
HF_HOME = ROOT / "data" / "hf"

DATASETS = [
    ("msra", ROOT / "data" / "hf" / "local" / "msra"),
    ("resume", ROOT / "data" / "hf" / "local" / "resume"),
    ("weibo", ROOT / "data" / "hf" / "local" / "weibo"),
    ("cross_ner_all", ROOT / "data" / "hf" / "local" / "cross_ner_all"),
]

# 默认超参数（保守设置以避免显存溢出 OOM）
BERT_MODEL = os.environ.get("EVAL_BERT_MODEL", "hfl/chinese-roberta-wwm-ext-large")
TRAIN_BS = int(os.environ.get("EVAL_TRAIN_BS", 2))
EVAL_BS = int(os.environ.get("EVAL_EVAL_BS", 8))
GRAD_ACC = int(os.environ.get("EVAL_GRAD_ACC", 4))
NUM_EPOCHS = float(os.environ.get("EVAL_NUM_EPOCHS", 1))
MAX_LENGTH = int(os.environ.get("EVAL_MAX_LENGTH", 128))

STAMP = time.strftime("%Y%m%d_%H%M%S")
RUN_ROOT = BASE / "runs" / "ner_eval" / STAMP
LOG_DIR = BASE / "logs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

summary = {}

# 工具函数：查找 run_eval 生成的最新结果目录（命名为 model-dataset-timestamp）
def _safe_name(s: str) -> str:
    s = s.replace(os.sep, "-").replace(" ", "_")
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "-" for c in s)

def _model_key(model: str) -> str:
    base = os.path.basename(model.rstrip("/")) if os.path.sep in model else model
    return _safe_name(base)

RESULTS_ROOT = BASE / "results"
MODEL_KEY = _model_key(BERT_MODEL)

for name, path in DATASETS:
    if not path.exists():
        print(f"[WARN] skip {name}, missing: {path}")
        continue

    env = os.environ.copy()
    env["HF_HOME"] = str(HF_HOME)
    env["HF_DATASETS_CACHE"] = str(HF_HOME / "datasets")
    env["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    env["HF_OFFLINE"] = env.get("HF_OFFLINE", "1")
    # 优先使用本地保存的数据集
    env["DATASET_PATH"] = str(path)
    env["BERT_MODEL"] = BERT_MODEL
    env["TRAIN_SAMPLES"] = "0"
    env["EVAL_SAMPLES"] = "0"
    env["NUM_EPOCHS"] = str(NUM_EPOCHS)
    env["TRAIN_BS"] = str(TRAIN_BS)
    env["EVAL_BS"] = str(EVAL_BS)
    env["GRAD_ACC"] = str(GRAD_ACC)
    env["MAX_LENGTH"] = str(MAX_LENGTH)
    # 指标选择（默认 entity）；允许由外部环境变量覆盖
    env["METRIC"] = env.get("METRIC", "entity")
    # 新增的行为开关
    env["MODEL_PAPER_ALIGNED"] = env.get("MODEL_PAPER_ALIGNED", "0")
    env["MEGA_LAPLACIAN"] = env.get("MEGA_LAPLACIAN", "0")
    env["DYNAMIC_PADDING"] = env.get("DYNAMIC_PADDING", "1")
    # 不设置 OUTPUT_DIR；run_eval 会写入 results/{model}-{dataset}-{ts}
    env["EVAL_SAVE"] = "0"  # 关闭保存以节省磁盘空间

    log_file = LOG_DIR / f"eval_{name}.log"
    print(f"[RUN] {name}")
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"=== Eval {name} @ {time.strftime('%F %T')} ===\n")
        lf.flush()
        proc = subprocess.run([PY, str(RUN_SCRIPT)], env=env, stdout=lf, stderr=subprocess.STDOUT)
        code = proc.returncode
        print(f"[DONE] {name} exit={code}")

    # 运行结束后，查找匹配前缀的最新目录并读取指标
    prefix = f"{MODEL_KEY}-{_safe_name(name)}-"
    candidates = [d for d in RESULTS_ROOT.glob(f"{prefix}*") if d.is_dir()]
    if not candidates:
        summary[name] = {"error": f"no result dir with prefix {prefix} under {RESULTS_ROOT}"}
    else:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        m_path = latest / "metrics_entity.json"
        try:
            if m_path.exists():
                with open(m_path, "r", encoding="utf-8") as f:
                    summary[name] = json.load(f)
            else:
                summary[name] = {"error": f"no metrics file: {m_path}"}
        except Exception as e:
            summary[name] = {"error": str(e)}

# 写出汇总
with open(RUN_ROOT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("All done. Summary at:", RUN_ROOT / "summary.json")

