import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(r"c:\Users\msherazi\BAN Auth Project\results09202025")

# Heuristics to drop large/noisy entries
DROP_KEYS = {
    "indices", "kept_frame_indices", "y_true", "y_pred", "probs", "probabilities",
    "roc_curve", "pr_curve", "cm", "confusion_matrix", "embeddings", "pca_components",
    # Also drop frame id-like fields
    "frame_id", "frame_ids", "frameindex", "frame_index", "frame_indices", "frameidx", "frameid",
}

# Column name patterns to drop from CSV/TSV tables (lowercased comparison)
NOISE_COL_PATTERNS = [
    r"^(?:index|unnamed:.*)$",           # pandas default/unnamed index columns
    r"^y_true$", r"^y_pred$",          # raw labels/preds
    r"^probs?$", r"^probabilities$",   # probability columns
    r"^frame_id$", r"^frame_ids$", r"^frameid$",
    r"^frame_index$", r"^frame_indices$", r"^frameindex$", r"^frame_idx$", r"^frameidx$",
]

# Metric / config key patterns to retain (lower-cased substring match)
METRIC_PATTERNS = [
    "acc", "accuracy", "f1", "auc", "eer", "ece", "brier", "log_loss",
    "macro", "micro", "weighted", "chance", "std", "mean", "avg", "support",
    "top1", "top-1",
]
MANIFEST_KEEP_PATTERNS = [
    "model", "classifier", "clf", "estimator", "algorithm", "params", "hyper",
    "cv", "fold", "split", "n_splits", "seed", "random_state",
    "calibration", "temperature", "temp", "ece",
    "dataset", "task", "window", "feature", "features", "classes", "num_",
]

# Limits to keep JSON concise
MAX_ROWS_PER_TABLE = 15
MAX_METRIC_ITEMS = 100


def is_large_list(v: Any, max_len: int = 500) -> bool:
    try:
        return isinstance(v, list) and len(v) > max_len
    except Exception:
        return False


def is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def key_matches(patterns: List[str], key: str) -> bool:
    kl = key.lower()
    return any(p in kl for p in patterns)


def clean_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if any(bad in kl for bad in DROP_KEYS):
                continue
            if kl.endswith("_indices") or kl.endswith("_index"):
                continue
            if is_large_list(v):
                continue
            out[k] = clean_obj(v)
        return out
    if isinstance(obj, list):
        if is_large_list(obj):
            return []
        return [clean_obj(v) for v in obj]
    return obj


def filter_metrics_dict(obj: Any, budget: List[int]) -> Any:
    """Recursively keep only metric-like scalar fields. `budget` is a single-item list used to cap total items."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if budget[0] <= 0:
                break
            if is_scalar(v) and key_matches(METRIC_PATTERNS, str(k)):
                out[k] = v
                budget[0] -= 1
            elif isinstance(v, dict):
                sub = filter_metrics_dict(v, budget)
                if sub:
                    out[k] = sub
            else:
                # skip lists and non-scalar non-dicts
                continue
        return out
    return obj if is_scalar(obj) else None


def shrink_manifest(man: Any) -> Dict[str, Any]:
    if not isinstance(man, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in man.items():
        if not key_matches(MANIFEST_KEEP_PATTERNS, str(k)):
            continue
        if is_scalar(v):
            out[k] = v
        # Keep small nested dicts as flattened scalars where possible
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if is_scalar(sv) and key_matches(MANIFEST_KEEP_PATTERNS, f"{k}.{sk}"):
                    out[f"{k}.{sk}"] = sv
    # Hard cap key count
    if len(out) > 40:
        # Keep first 40 deterministic by sorted keys
        pruned = {}
        for k in sorted(out.keys())[:40]:
            pruned[k] = out[k]
        return pruned
    return out


def pick_metric_columns(rows: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for r in rows:
        keys.update(r.keys())
    cand = [k for k in keys if key_matches(
        [
            "device", "dev", "id", "position", "pos", "class", "label",
            "acc", "accuracy", "f1", "auc", "eer", "ece", "support", "n", "count", "samples",
            "macro", "micro", "weighted", "logo", "fold",
        ],
        k,
    )]
    # Preserve a stable order: important id columns first then metrics
    id_cols = [c for c in ["device", "device_id", "dev", "id", "position", "class", "label"] if c in cand]
    metric_cols = [c for c in [
        "accuracy", "acc", "f1", "auc", "eer", "ece", "support", "n", "count", "samples",
        "macro_acc", "macro_f1", "macro_auc", "micro_acc", "micro_f1", "weighted_acc", "weighted_f1",
        "logo", "fold",
    ] if c in cand]
    keep = id_cols + metric_cols
    # Fallback: if empty, keep up to 5 columns from original
    if not keep:
        keep = list(keys)[:5]
    return keep


def prune_table(rows: List[Dict[str, Any]], max_rows: int = MAX_ROWS_PER_TABLE) -> List[Dict[str, Any]]:
    if not rows:
        return []
    keep_cols = pick_metric_columns(rows)
    pruned: List[Dict[str, Any]] = []
    for r in rows[:max_rows]:
        pruned.append({k: r.get(k, None) for k in keep_cols})
    return pruned


def load_json_safe(p: Path) -> Dict[str, Any]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def drop_noise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop index/frame/probability-like noisy columns from a dataframe."""
    try:
        cols_to_drop: List[str] = []
        for c in df.columns:
            cl = str(c).lower()
            # Generic suffix-based indices
            if cl.endswith("_indices") or cl.endswith("_index"):
                cols_to_drop.append(c)
                continue
            for pat in NOISE_COL_PATTERNS:
                if re.search(pat, cl):
                    cols_to_drop.append(c)
                    break
        if cols_to_drop:
            df = df.drop(columns=list(set(cols_to_drop)), errors="ignore")
        return df
    except Exception:
        return df


def load_csv_row_dict(p: Path) -> List[Dict[str, Any]]:
    try:
        # Set separator based on extension to avoid parser warnings
        ext = p.suffix.lower()
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(p, sep=sep, engine="python")
        df = drop_noise_columns(df)
        # Convert to list of row dicts with basic Python types
        rows = []
        for _, row in df.iterrows():
            rec: Dict[str, Any] = {}
            for c, v in row.items():
                if pd.isna(v):
                    rec[c] = None
                else:
                    # Cast numpy types to Python primitives
                    if hasattr(v, "item"):
                        try:
                            v = v.item()
                        except Exception:
                            v = str(v)
                    rec[c] = v
            rows.append(rec)
        return rows
    except Exception:
        return []


def list_files_with_ext(folder: Path, exts: List[str]) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if any(fn.lower().endswith(e.lower()) for e in exts):
                out.append(str(Path(root) / fn))
    return out


def latest_run_dirs(base: Path, prefix: str) -> List[Path]:
    # Return all run dirs sorted by embedded timestamp desc
    runs = []
    for child in base.glob(f"{prefix}__*"):
        if child.is_dir():
            m = re.search(r"__(\w+)_(\d+)$", child.name)
            if m:
                ts = int(m.group(2))
            else:
                # fallback: extract last number group
                m2 = re.search(r"(\d+)$", child.name)
                ts = int(m2.group(1)) if m2 else 0
            runs.append((ts, child))
    runs.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in runs]


# Note: we intentionally drop verbose text_data to keep JSON concise

def collect_onbody_device_identification(folder: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"artifacts": {}}
    res_json = folder / "onbody_device_identification_results.json"
    if res_json.exists():
        raw = clean_obj(load_json_safe(res_json))
        out["raw_results"] = filter_metrics_dict(raw, [MAX_METRIC_ITEMS]) or {}
    # models and figures in root
    model_paths = [str(p) for p in folder.glob("*best_model*.joblib")]
    figs = {
        "feature_importance": str(folder / "feature_importance.png"),
        "reliability_png": str(folder / "reliability.png"),
        "reliability_bins_csv": str(folder / "reliability_bins.csv"),
        "pca_png": str(folder / "pca.png"),
        "confusion_matrix_png": str(folder / "confusion_matrix.png"),
        "confusion_matrix_csv": str(folder / "confusion_matrix.csv"),
    }
    out["artifacts"] = {**{k: v for k, v in figs.items() if v and Path(v).exists()}, "models": model_paths}
    # Only keep latest run manifest/paths
    runs = latest_run_dirs(folder, "onbody_device_identification")
    if runs:
        r = runs[0]
        run_entry = {
            "run_dir": str(r),
            "calibration_sweep_csv": str(r / "calibration_sweep.csv") if (r / "calibration_sweep.csv").exists() else None,
            "separability_csv": str(r / "separability.csv") if (r / "separability.csv").exists() else None,
            "split_diagnostics_csv": str(r / "split_diagnostics.csv") if (r / "split_diagnostics.csv").exists() else None,
            "manifest": shrink_manifest(load_json_safe(r / "data_pack.json")) if (r / "data_pack.json").exists() else None,
        }
        out["latest_run"] = run_entry
    return out


def collect_offbody_verification(folder: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"artifacts": {}}
    res_json = folder / "offbody_verification_results.json"
    if res_json.exists():
        raw = clean_obj(load_json_safe(res_json))
        out["raw_results"] = filter_metrics_dict(raw, [MAX_METRIC_ITEMS]) or {}
    runs = latest_run_dirs(folder, "offbody_verification")
    if runs:
        r = runs[0]
        summ = r / "verification_summary.csv"
        devices_dir = r / "devices"
        per_device = {}
        if devices_dir.exists():
            for d in sorted([p for p in devices_dir.iterdir() if p.is_dir()]):
                dev_id = d.name.replace("device_", "")
                per_device[dev_id] = {
                    "roc_png": str(d / "roc.png") if (d / "roc.png").exists() else None,
                    "roc_csv": str(d / "roc.csv") if (d / "roc.csv").exists() else None,
                    "thresholds_json": str(d / "thresholds.json") if (d / "thresholds.json").exists() else None,
                }
        out["latest_run"] = {
            "run_dir": str(r),
            "summary_csv": str(summ) if summ.exists() else None,
            "per_device_artifacts": per_device,
            "manifest": shrink_manifest(load_json_safe(r / "data_pack.json")) if (r / "data_pack.json").exists() else None,
        }
    return out


def collect_device_identification_combined(folder: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    res_json = folder / "device_identification_combined_results.json"
    if res_json.exists():
        raw = clean_obj(load_json_safe(res_json))
        out["raw_results"] = filter_metrics_dict(raw, [MAX_METRIC_ITEMS]) or {}
    return out


def collect_offbody_position(folder: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"artifacts": {}}
    glob_sum = folder / "offbody_position_global_summary.csv"
    per_dev = folder / "per_device_summary.csv"
    if glob_sum.exists():
        out["global_summary_rows"] = prune_table(load_csv_row_dict(glob_sum), max_rows=5)
    if per_dev.exists():
        out["per_device_rows"] = prune_table(load_csv_row_dict(per_dev), max_rows=MAX_ROWS_PER_TABLE)
    # model and figures
    model_paths = [str(p) for p in folder.glob("*global_best_model*.joblib")]
    figs = {
        "feature_importance": str(folder / "feature_importance.png"),
        "reliability_png": str(folder / "reliability.png"),
        "reliability_bins_csv": str(folder / "reliability_bins.csv"),
        "pca_png": str(folder / "pca.png"),
        "confusion_matrix_png": str(folder / "confusion_matrix.png"),
        "confusion_matrix_csv": str(folder / "confusion_matrix.csv"),
    }
    out["artifacts"] = {**{k: v for k, v in figs.items() if v and Path(v).exists()}, "models": model_paths}
    # Keep only latest run manifest
    runs = latest_run_dirs(folder, "offbody_position")
    if runs:
        r = runs[0]
        out["latest_run"] = {
            "run_dir": str(r),
            "manifest": shrink_manifest(load_json_safe(r / "data_pack.json")) if (r / "data_pack.json").exists() else None,
        }
    return out


def collect_onbody_movement(folder: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"artifacts": {}}
    glob_sum = folder / "onbody_movement_global_summary.csv"
    if glob_sum.exists():
        out["global_summary_rows"] = prune_table(load_csv_row_dict(glob_sum), max_rows=5)
    # figures if any
    figs = {
        "reliability_png": str(folder / "reliability.png"),
        "reliability_bins_csv": str(folder / "reliability_bins.csv"),
        "confusion_matrix_png": str(folder / "confusion_matrix.png"),
        "confusion_matrix_csv": str(folder / "confusion_matrix.csv"),
        "roc_png": str(folder / "roc.png"),
        "roc_csv": str(folder / "roc.csv"),
    }
    out["artifacts"] = {k: v for k, v in figs.items() if v and Path(v).exists()}
    runs = latest_run_dirs(folder, "onbody_movement")
    if runs:
        r = runs[0]
        out["latest_run"] = {
            "run_dir": str(r),
            "manifest": shrink_manifest(load_json_safe(r / "data_pack.json")) if (r / "data_pack.json").exists() else None,
        }
    return out


def collect_all_analyses(folder: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    res_json = folder / "all_analyses_results.json"
    if res_json.exists():
        raw = clean_obj(load_json_safe(res_json))
        out["raw_results"] = filter_metrics_dict(raw, [MAX_METRIC_ITEMS]) or {}
    return out


def collect_hierarchical(folder: Path) -> Dict[str, Any]:
    # Omit verbose hierarchical details to keep JSON concise (manuscript skips these metrics)
    return {"note": "hierarchical cascade metrics omitted for concision"}


def collect_mixed_scenario(folder: Path) -> Dict[str, Any]:
    """Collect concise mixed scenario results and latest run artifacts for baseline/robust/leaky."""
    out: Dict[str, Any] = {"variants": {}, "artifacts": {}}
    # Top-level aggregated results (baseline/robust/leaky) -> keep only metric-like scalars via filter
    res_json = folder / "mixed_scenario_results.json"
    if res_json.exists():
        raw = clean_obj(load_json_safe(res_json))
        out["raw_results"] = filter_metrics_dict(raw, [MAX_METRIC_ITEMS]) or {}
    # For each variant, capture latest run paths for calibration/separability/split diagnostics and manifest
    def variant_latest(variant: str) -> Dict[str, Any]:
        vdir = folder / variant
        if not vdir.exists():
            return {"note": "variant folder not found"}
        # Prefix is like mixed_scenario_<variant>
        prefix = f"mixed_scenario_{variant}"
        runs = latest_run_dirs(vdir, prefix)
        if not runs:
            return {"note": "no runs found"}
        r = runs[0]
        entry: Dict[str, Any] = {
            "run_dir": str(r),
            "calibration_sweep_csv": str(r / "calibration_sweep.csv") if (r / "calibration_sweep.csv").exists() else None,
            "separability_csv": str(r / "separability.csv") if (r / "separability.csv").exists() else None,
            "split_diagnostics_csv": str(r / "split_diagnostics.csv") if (r / "split_diagnostics.csv").exists() else None,
            "manifest": shrink_manifest(load_json_safe(r / "data_pack.json")) if (r / "data_pack.json").exists() else None,
        }
        return entry
    for variant in ["baseline", "robust", "leaky"]:
        out["variants"][variant] = variant_latest(variant)
    return out


def main():
    unified: Dict[str, Any] = {"root": str(ROOT), "generated_by": "build_unified_results.py", "version": 2}

    # Map analysis folders to collectors
    mapping = [
        ("onbody_device_identification", collect_onbody_device_identification),
        ("offbody_verification", collect_offbody_verification),
        ("device_identification_combined", collect_device_identification_combined),
        ("offbody_position", collect_offbody_position),
        ("onbody_movement", collect_onbody_movement),
        ("all_analyses", collect_all_analyses),
        ("hierarchical_classification", collect_hierarchical),
        ("mixed_scenario", collect_mixed_scenario),
    ]

    for name, fn in mapping:
        folder = ROOT / name
        if folder.exists() and folder.is_dir():
            try:
                unified[name] = fn(folder)
            except Exception as e:
                unified[name] = {"error": str(e)}
        else:
            unified[name] = {"note": "folder not found"}

    # Also include any stray results file names in root (paths only, not contents)
    extras: Dict[str, Any] = {"json_results": [], "csv_summaries": []}
    for p in ROOT.glob("*.json"):
        if p.name.endswith("_results.json") and p.name not in [
            "onbody_device_identification_results.json",
            "offbody_verification_results.json",
            "device_identification_combined_results.json",
            "all_analyses_results.json",
        ]:
            extras["json_results"].append(str(p))
    for p in ROOT.glob("*_summary.csv"):
        extras["csv_summaries"].append(str(p))
    if extras["json_results"] or extras["csv_summaries"]:
        unified["extras"] = extras

    out_path = ROOT / "unified_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2)
    print(f"Unified results written to: {out_path}")


if __name__ == "__main__":
    main()
