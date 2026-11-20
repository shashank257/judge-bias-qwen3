"""Paths, seed, dataset records and version log shared by all notebooks."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

SEED = 261
PIPELINE_VERSION = "week1-v1.0"

DATASETS = {
    # key: (hf repo, pinned revision, license, files)
    "arena2023_official": ("lmsys/chatbot_arena_conversations", "1b6335d42a1d2c7e34870c905d03ab964f7f2bd8",
                           "CC-BY-4.0 (conversations) / CC-BY-NC-4.0 (gated; user agreement)",
                           ["data/train-00000-of-00001-cced8514c7ed782a.parquet"]),
    "arena2023_mirror": ("agie-ai/lmsys-chatbot_arena_conversations", "d45bf836ec5a09d8a98881b4ae16347ffb6aae9b",
                         "mirror of lmsys/chatbot_arena_conversations (same terms)",
                         ["data/train-00000-of-00001-f4a410a7274a9509.parquet"]),
    "arena2025": ("lmarena-ai/arena-human-preference-140k", "6322995ab34d7c2693e3f47dd13fa5caa0789a74", "CC-BY-4.0",
                  [f"data/train-0000{i}-of-00007.parquet" for i in range(7)]),
    "mtbench_human": ("lmsys/mt_bench_human_judgments", "f7d2896d2cc5d80f8b55c2bbc722613555233c25", "CC-BY-4.0",
                      ["data/human-00000-of-00001-25f4910818759289.parquet"]),
    "ultrafeedback": ("openbmb/UltraFeedback", "40b436560ca83a8dba36114c22ab3c66e43f6d5e", "MIT",
                      ["sharegpt.jsonl", "ultrachat.jsonl", "evol_instruct.jsonl"]),
}


def project_root() -> str:
    return os.environ.get("DSC261_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def path(*parts) -> str:
    p = os.path.join(project_root(), *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(key: str, token: str | None = None) -> list[str]:
    from huggingface_hub import hf_hub_download
    repo, rev, _, files = DATASETS[key]
    return [hf_hub_download(repo, f, repo_type="dataset", revision=rev, token=token,
                            local_dir=path("data", "raw", key)) for f in files]


def record_dataset(key: str, files: list[str], n_rows: int, note: str = "") -> dict:
    """Append/replace this dataset's row in data/dataset_records.json."""
    repo, rev, lic, _ = DATASETS[key]
    rec = {"key": key, "hf_repo": repo, "revision": rev, "license": lic,
           "files": [{"path": os.path.relpath(f, project_root()), "sha256": sha256_file(f),
                      "bytes": os.path.getsize(f)} for f in files],
           "n_rows": int(n_rows), "downloaded_utc": dt.datetime.utcnow().strftime("%Y-%m-%d"), "note": note}
    p = path("data", "dataset_records.json")
    recs = json.load(open(p)) if os.path.exists(p) else {}
    recs[key] = rec
    json.dump(recs, open(p, "w"), indent=2)
    return rec


def log_version(step: str, detail: str, outputs: list[str] | None = None):
    """Preprocessing version log (append-only JSONL)."""
    row = {"utc": dt.datetime.utcnow().isoformat(timespec="seconds"), "pipeline": PIPELINE_VERSION,
           "seed": SEED, "step": step, "detail": detail,
           "outputs": {os.path.relpath(o, project_root()): sha256_file(o)[:16] for o in (outputs or [])}}
    with open(path("data", "version_log.jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row
