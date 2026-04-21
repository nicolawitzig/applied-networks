from pathlib import Path
import json
import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config/config.yaml") -> dict:
    cfg_path = PROJECT_ROOT / path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def write_jsonl(rows, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(resolve_path(path))
