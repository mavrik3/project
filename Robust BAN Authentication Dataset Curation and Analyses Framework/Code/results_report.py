import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _collect_latest_pack(dir_path: Path, prefix: str) -> Optional[Path]:
    if not dir_path.exists():
        return None
    candidates = sorted([p for p in dir_path.iterdir() if p.is_dir() and p.name.startswith(prefix + "__")])
    return candidates[-1] if candidates else None


def build_manuscript_summary(results_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # all_analyses summary
    all_dir = results_root / "all_analyses"
    all_csv = all_dir / "all_analyses_summary.csv"
    df_all = _read_csv_safe(all_csv)
    if df_all is not None and len(df_all) > 0:
        out["all_analyses_summary_csv"] = str(all_csv)
        out["all_analyses_summary_row"] = df_all.iloc[0].to_dict()

    # hierarchical
    hier_dir = results_root / "hierarchical_classification"
    latest_hier = _collect_latest_pack(hier_dir, "hierarchical_classification")
    if latest_hier:
        hier_csv = latest_hier / "hierarchical_summary.csv"
        df_h = _read_csv_safe(hier_csv)
        if df_h is not None and len(df_h) > 0:
            out["hierarchical_summary_csv"] = str(hier_csv)
            out["hierarchical_summary_row"] = df_h.iloc[0].to_dict()

    # verification
    ver_dir = results_root / "offbody_verification"
    latest_ver = _collect_latest_pack(ver_dir, "offbody_verification")
    if latest_ver:
        ver_csv = latest_ver / "verification_summary.csv"
        df_v = _read_csv_safe(ver_csv)
        if df_v is not None and len(df_v) > 0:
            # macro row is first by design
            out["verification_summary_csv"] = str(ver_csv)
            out["verification_macro_row"] = df_v.iloc[0].to_dict()

    # offbody position
    pos_dir = results_root / "offbody_position"
    pos_global_csv = pos_dir / "offbody_position_global_summary.csv"
    df_pg = _read_csv_safe(pos_global_csv)
    if df_pg is not None and len(df_pg) > 0:
        out["offbody_position_global_summary_csv"] = str(pos_global_csv)
        out["offbody_position_global_row"] = df_pg.iloc[0].to_dict()
    per_dev_csv = pos_dir / "per_device_summary.csv"
    df_pd = _read_csv_safe(per_dev_csv)
    if df_pd is not None and not df_pd.empty:
        out["offbody_position_per_device_summary_csv"] = str(per_dev_csv)

    # device identification (on/off)
    on_dev_csv = results_root / "onbody_device_identification" / "onbody_device_identification_summary.csv"
    df_od = _read_csv_safe(on_dev_csv)
    if df_od is not None and len(df_od) > 0:
        out["onbody_device_identification_summary_csv"] = str(on_dev_csv)
        out["onbody_device_identification_row"] = df_od.iloc[0].to_dict()
    off_dev_csv = results_root / "offbody_device_identification" / "offbody_device_identification_summary.csv"
    df_offd = _read_csv_safe(off_dev_csv)
    if df_offd is not None and len(df_offd) > 0:
        out["offbody_device_identification_summary_csv"] = str(off_dev_csv)
        out["offbody_device_identification_row"] = df_offd.iloc[0].to_dict()

    # onbody movement (global)
    mov_dir = results_root / "onbody_movement"
    mov_csv = mov_dir / "onbody_movement_global_summary.csv"
    df_mv = _read_csv_safe(mov_csv)
    if df_mv is not None and len(df_mv) > 0:
        out["onbody_movement_global_summary_csv"] = str(mov_csv)
        out["onbody_movement_global_row"] = df_mv.iloc[0].to_dict()

    # mixed scenario
    mix_dir = results_root / "mixed_scenario"
    for mode in ("baseline", "robust", "leaky"):
        mdir = mix_dir / mode
        msum = mdir / f"mixed_scenario_{mode}_summary.csv"
        df_m = _read_csv_safe(msum)
        if df_m is not None and len(df_m) > 0:
            out[f"mixed_{mode}_summary_csv"] = str(msum)
            out[f"mixed_{mode}_summary_row"] = df_m.iloc[0].to_dict()

    # offbody device-id per position
    opd_dir = results_root / "offbody_deviceid_per_position"
    perpos_csv = opd_dir / "per_position_summary.csv"
    df_pp = _read_csv_safe(perpos_csv)
    if df_pp is not None and not df_pp.empty:
        out["offbody_deviceid_per_position_summary_csv"] = str(perpos_csv)
        # compute macro if not present elsewhere
        try:
            if "accuracy" in df_pp.columns:
                out["offbody_deviceid_per_position_macro_acc"] = float(pd.to_numeric(df_pp["accuracy"], errors="coerce").dropna().mean())
        except Exception:
            pass

    # Write combined artifacts
    manus_dir = results_root / "manuscript"
    manus_dir.mkdir(parents=True, exist_ok=True)

    # Flatten to a single-row CSV for quick copy into tables
    flat_row = {}
    for k, v in out.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat_row[f"{k}.{kk}"] = vv
        else:
            flat_row[k] = v

    combined_csv = manus_dir / "combined_summary.csv"
    pd.DataFrame([flat_row]).to_csv(combined_csv, index=False)

    combined_json = manus_dir / "combined_summary.json"
    with open(combined_json, "w") as f:
        json.dump(out, f, indent=2)

    return {
        "combined_summary_csv": str(combined_csv),
        "combined_summary_json": str(combined_json),
        **out,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate results for manuscript tables/figures.")
    parser.add_argument("--results-root", type=str, default="C:/Users/msherazi/BAN Auth Project/results09202025",
                        help="Path to results root (e.g., results09202025)")
    args = parser.parse_args()

    root = Path(args.results_root)
    summary = build_manuscript_summary(root)
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("_csv") or k.endswith("_json")}, indent=2))


if __name__ == "__main__":
    main()
