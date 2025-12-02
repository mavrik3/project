from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Sequence, Any
import logging, json, re
import pandas as pd
import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHANNEL_CANDIDATES = ["channel", "chan", "ch", "freq", "frequency", "fhss_channel"]
SCENARIO_CANDIDATES = ["scenario", "test", "scenario_label"]
POSITION_CANDIDATES = ["pos_label", "position", "pos"]
DEVICE_CANDIDATES = ["dvc", "device", "device_id"]
IQ_CANDIDATES = ["frame", "iq", "iq_frame", "iq_data", "raw_iq"]


class DataLoader:
    """Utility for loading raw dataset artifacts from disk."""

    @staticmethod
    def list_available_datasets(root: Path) -> List[Path]:
        root = Path(root)
        if not root.exists():
            return []
        out: List[Path] = []
        for p in root.iterdir():
            if not p.is_dir():
                continue
            if (p / "index.parquet").exists() or (p / "index.csv").exists():
                out.append(p)
        return sorted(out)

    @staticmethod
    def load_dataset(dir_path: Path) -> Tuple[pd.DataFrame, dict]:
        dir_path = Path(dir_path)
        idx_p = dir_path / "index.parquet"
        card_p = dir_path / "card.json"
        df: pd.DataFrame
        if idx_p.exists():
            try:
                df = pd.read_parquet(idx_p)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed reading parquet {idx_p} ({e}); trying CSV fallback.")
                csv_p = dir_path / "index.csv"
                if csv_p.exists():
                    df = pd.read_csv(csv_p)
                else:
                    df = pd.DataFrame()
        else:
            csv_p = dir_path / "index.csv"
            if csv_p.exists():
                df = pd.read_csv(csv_p)
            else:
                df = pd.DataFrame()
        card = {}
        if card_p.exists():
            try:
                card = json.loads(card_p.read_text())
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to read card.json: {e}")
        return df, card

    @staticmethod
    def load_from_pickle(
        pkl_path: Union[str, Path],
        preproc_path: Optional[Union[str, Path]] = None,
        force_recompute: bool = False,
        debug_toy_subset: bool = False,
    ) -> pd.DataFrame:
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            logger.warning(f"Pickle path not found: {pkl_path}")
            return pd.DataFrame()
        df = pd.read_pickle(pkl_path)
        if not isinstance(df, pd.DataFrame):  # pragma: no cover
            logger.warning("Loaded pickle is not a DataFrame.")
            return pd.DataFrame()
        if debug_toy_subset and len(df) > 2000:
            df = df.sample(2000, random_state=42).reset_index(drop=True)
        return df.reset_index(drop=True)

    @staticmethod
    def load_offbody_fallback(
        unified_pkl_path: Union[str, Path], max_rows: Optional[int] = None
    ) -> pd.DataFrame:
        df = DataLoader.load_from_pickle(unified_pkl_path)
        if df.empty:
            return df
        scen_col = DataPreprocessor.detect_column(df, SCENARIO_CANDIDATES)
        if scen_col:
            df = df[df[scen_col].astype(str).str.lower() == "offbody"].copy()
        if max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows).copy()
        return df.reset_index(drop=True)


class DataPreprocessor:
    """Column normalization & light IQ preprocessing (optional)."""

    @staticmethod
    def detect_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        # fuzzy: lowercase match
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        # channel
        ch_col = DataPreprocessor.detect_column(df, CHANNEL_CANDIDATES)
        if ch_col and ch_col != "channel":
            df.rename(columns={ch_col: "channel"}, inplace=True)
        # scenario
        scen_col = DataPreprocessor.detect_column(df, SCENARIO_CANDIDATES)
        if scen_col and scen_col != "scenario":
            df.rename(columns={scen_col: "scenario"}, inplace=True)
        # device
        dvc_col = DataPreprocessor.detect_column(df, DEVICE_CANDIDATES)
        if dvc_col and dvc_col != "dvc":
            df.rename(columns={dvc_col: "dvc"}, inplace=True)
        # position
        pos_col = DataPreprocessor.detect_column(df, POSITION_CANDIDATES)
        if pos_col and pos_col != "pos_label":
            df.rename(columns={pos_col: "pos_label"}, inplace=True)
        # ensure channel int
        if "channel" in df.columns:
            try:
                df["channel"] = df["channel"].astype(int)
            except Exception:
                pass
        # global frame id fallback
        if "global_frame_id" not in df.columns:
            df["global_frame_id"] = np.arange(len(df), dtype=np.int64)
        return df

    @staticmethod
    def detect_iq_column(df: pd.DataFrame) -> Optional[str]:
        for c in IQ_CANDIDATES:
            if c in df.columns:
                # quick heuristic: object or list-like entries
                if df[c].dtype == object:
                    return c
        return None

    @staticmethod
    def is_valid_iq(arr) -> bool:
        if arr is None:
            return False
        if isinstance(arr, (list, tuple)) and len(arr) > 4:
            return True
        if isinstance(arr, np.ndarray) and arr.ndim <= 2 and arr.size > 4:
            return True
        return False

    @staticmethod
    def to_complex(arr) -> Optional[np.ndarray]:
        if isinstance(arr, np.ndarray):
            if np.iscomplexobj(arr):
                return arr.astype(np.complex64)
            # real/imag pair
            if arr.ndim == 2 and arr.shape[-1] == 2:
                return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
        if isinstance(arr, (list, tuple)):
            a = np.asarray(arr)
            if np.iscomplexobj(a):
                return a.astype(np.complex64)
            if a.ndim == 2 and a.shape[-1] == 2:
                return (a[:, 0] + 1j * a[:, 1]).astype(np.complex64)
        return None

    @staticmethod
    def extract_preamble(
        x: np.ndarray,
        preamble_len_bits: int = 8,
        candidate_sps: Optional[Sequence[int]] = None,
        min_score: float = 0.25,
    ) -> Optional[np.ndarray]:
        # Placeholder heuristic: just return first N samples if length ok.
        if x.size < preamble_len_bits * 2:
            return None
        return x[: preamble_len_bits * 2]

    @staticmethod
    def normalize_iq_signal(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        xr = x - np.mean(x)
        std = np.std(xr)
        if std > 0:
            xr = xr / std
        return xr

    @staticmethod
    def trim_transient(x: np.ndarray, drop_min: int = 800, frac: float = 0.08) -> np.ndarray:
        if x.size <= drop_min:
            return x
        k = int(min(len(x) * frac, drop_min))
        return x[k:]

    @staticmethod
    def preprocess_iq_frames(df: pd.DataFrame, target_length: int = 2000) -> pd.DataFrame:
        iq_col = DataPreprocessor.detect_iq_column(df)
        if iq_col is None:
            return df
        out = []
        for v in df[iq_col]:
            if not DataPreprocessor.is_valid_iq(v):
                out.append(None)
                continue
            c = DataPreprocessor.to_complex(v)
            if c is None:
                out.append(None)
                continue
            c = DataPreprocessor.normalize_iq_signal(c)
            c = DataPreprocessor.trim_transient(c)
            if c.size > target_length:
                c = c[:target_length]
            out.append(c)
        df = df.copy()
        df["iq_proc"] = out
        return df

    @staticmethod
    def prepare_offbody_dataframe(df: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
        if df.empty:
            return df
        if "scenario" in df.columns:
            m = df["scenario"].astype(str).str.lower() == "offbody"
            df = df[m].copy()
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows).copy()
        return df.reset_index(drop=True)

    @staticmethod
    def prepare_onbody_dataframe(df: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
        if df.empty:
            return df
        if "scenario" in df.columns:
            m = df["scenario"].astype(str).str.lower() == "onbody"
            df = df[m].copy()
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows).copy()
        return df.reset_index(drop=True)


class UnifiedDataManager:
    """High-level interface for dataset access & light preprocessing."""

    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        self.root_dir = Path(root_dir) if root_dir else None
        self.data_cache: Dict[str, pd.DataFrame] = {}

    # ----------------- Spec Parsing -----------------
    def _parse_source_spec(
        self, source_path: Optional[Union[str, Path, Dict[str, Union[str, Path]]]]
    ) -> Dict[str, Path]:
        spec: Dict[str, Path] = {}
        if source_path is None:
            if self.root_dir and (self.root_dir / "unified.pkl").exists():
                spec["unified"] = self.root_dir / "unified.pkl"
            return spec
        if isinstance(source_path, dict):
            for k, v in source_path.items():
                spec[k] = Path(v)
            return spec
        p = Path(source_path)
        if p.is_file() and p.suffix.lower() in {".pkl", ".pickle"}:
            name = p.stem.lower()
            if "unified" in name:
                spec["unified"] = p
            elif "onbody" in name:
                spec["onBody"] = p
            elif "offbody" in name:
                spec["offBody"] = p
            else:
                spec["unified"] = p  # fallback
        elif p.is_dir():
            # directory dataset index
            dir_name = p.name.lower()
            if "onbody" in dir_name:
                spec["onBody"] = p
            elif "offbody" in dir_name:
                spec["offBody"] = p
            else:
                # treat as unified root maybe containing unified.pkl
                if (p / "unified.pkl").exists():
                    spec["unified"] = p / "unified.pkl"
                else:
                    # attempt both inside datasets/
                    on_cand = p / "datasets" / "onbody_iqbb_hybrid"
                    off_cand = p / "datasets" / "offbody_iqbb_hybrid"
                    if on_cand.exists():
                        spec["onBody"] = on_cand
                    if off_cand.exists():
                        spec["offBody"] = off_cand
        return spec

    def _infer_sibling_dataset(self, path: Path, target: str) -> Optional[Path]:
        try:
            parent = path.parent
            if parent.name.lower() == "datasets":
                sib = parent / ("onbody_iqbb_hybrid" if target == "onBody" else "offbody_iqbb_hybrid")
                if sib.is_dir() and ((sib / "index.parquet").exists() or (sib / "index.csv").exists()):
                    return sib
        except Exception:  # pragma: no cover
            pass
        return None

    # ----------------- Preflight -----------------
    def preflight_check(
        self,
        scenario_filter: Optional[str] = None,
        source_path: Optional[Union[str, Path, Dict[str, Union[str, Path]]]] = None,
    ) -> Dict[str, Any]:
        exists: List[str] = []
        missing: List[str] = []
        spec = self._parse_source_spec(source_path)
        if not spec:
            return {"ok": False, "spec": {}, "exists": [], "missing": [], "message": "No sources resolved"}
        # function
        def _check(path: Path, kind: str):
            cond = False
            if kind == "dataset_dir":
                cond = path.is_dir() and ((path / "index.parquet").exists() or (path / "index.csv").exists())
            elif kind == "file":
                cond = path.is_file()
            if cond:
                exists.append(str(path))
            else:
                missing.append(str(path))
            return cond
        ok = True
        if scenario_filter in ("onBody", "offBody"):
            # need matching entry
            target = spec.get(scenario_filter)
            if target is None:
                # try unified fallback
                target = spec.get("unified")
            if target is None:
                ok = False
            else:
                if target.is_dir():
                    ok &= _check(target, "dataset_dir")
                else:
                    ok &= _check(target, "file")
        else:
            # just verify all present
            for k, p in spec.items():
                if p.is_dir():
                    ok &= _check(p, "dataset_dir")
                else:
                    ok &= _check(p, "file")
        return {
            "ok": bool(ok),
            "spec": {k: str(v) for k, v in spec.items()},
            "exists": sorted(set(exists)),
            "missing": sorted(set(missing)),
            "message": "OK" if ok else "Missing assets",
        }

    # ----------------- Load & Preprocess -----------------
    def load_and_preprocess(
        self,
        source_path: Optional[Union[str, Path, Dict[str, Union[str, Path]]]] = None,
        preprocess_iq: bool = False,
        scenario_filter: Optional[str] = None,
        max_rows: Optional[int] = None,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        spec = self._parse_source_spec(source_path)
        pf = self.preflight_check(scenario_filter, source_path)
        if not pf["ok"]:
            logger.warning(f"Preflight failed: {pf['message']}")
        # choose source
        chosen: Optional[Path] = None
        if scenario_filter in ("onBody", "offBody"):
            chosen = spec.get(scenario_filter) or spec.get("unified")
        else:
            # prefer unified else first entry
            chosen = spec.get("unified") or next(iter(spec.values()))
        if chosen is None:
            return pd.DataFrame()
        cache_key = f"{chosen}_{scenario_filter}_{max_rows}_{preprocess_iq}"
        if cache_key in self.data_cache and not force_reload:
            return self.data_cache[cache_key].copy()
        # load
        if chosen.is_dir():
            df, _ = DataLoader.load_dataset(chosen)
        elif chosen.suffix.lower() in {".pkl", ".pickle"}:
            df = DataLoader.load_from_pickle(chosen)
        else:
            logger.warning(f"Unsupported source type: {chosen}")
            return pd.DataFrame()
        if df.empty:
            logger.warning("Loaded dataframe is empty.")
            return df
        df = DataPreprocessor.standardize_columns(df)
        # scenario filter
        if scenario_filter and "scenario" in df.columns:
            df = df[df["scenario"].astype(str).str.lower() == scenario_filter.lower()].copy()
        # downsample / head
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows).copy()
        # IQ preprocess optional
        if preprocess_iq:
            df = DataPreprocessor.preprocess_iq_frames(df)
        # ensure channel column present
        if "channel" not in df.columns:
            # try derive from pattern inside maybe file name? fallback random removal -> skip
            raise ValueError("No 'channel' column found after standardization.")
        df = df.sort_values("global_frame_id").reset_index(drop=True)
        self.data_cache[cache_key] = df
        return df.copy()

    # convenience scenario-specific
    def get_scenario_dataframe(
        self,
        scenario: str,
        source_path: Optional[Union[str, Path, Dict[str, Union[str, Path]]]] = None,
        preprocess_iq: bool = False,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        return self.load_and_preprocess(
            source_path=source_path,
            preprocess_iq=preprocess_iq,
            scenario_filter=scenario,
            max_rows=max_rows,
        )

    def get_device_dataframe(
        self,
        device_id: Union[int, List[int]],
        scenario: Optional[str] = None,
        source_path: Optional[Union[str, Path]] = None,
        preprocess_iq: bool = False,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        df = self.load_and_preprocess(
            source_path=source_path,
            scenario_filter=scenario,
            preprocess_iq=preprocess_iq,
            max_rows=max_rows,
        )
        if df.empty or "dvc" not in df.columns:
            return pd.DataFrame()
        ids = [device_id] if isinstance(device_id, int) else list(device_id)
        return df[df["dvc"].isin(ids)].copy()

    def get_position_dataframe(
        self,
        position: Union[str, List[str]],
        scenario: Optional[str] = None,
        source_path: Optional[Union[str, Path]] = None,
        preprocess_iq: bool = False,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        df = self.load_and_preprocess(
            source_path=source_path,
            scenario_filter=scenario,
            preprocess_iq=preprocess_iq,
            max_rows=max_rows,
        )
        if df.empty or "pos_label" not in df.columns:
            return pd.DataFrame()
        pos = [position] if isinstance(position, str) else list(position)
        return df[df["pos_label"].isin(pos)].copy()

    def get_mixed_scenario_dataframe(
        self,
        source_path: Optional[Union[str, Path, Dict[str, Union[str, Path]]]] = None,
        device_ids: Optional[List[int]] = None,
        preprocess_iq: bool = False,
        max_per_scenario: Optional[int] = None,
    ) -> pd.DataFrame:
        # load both scenarios if available
        spec = self._parse_source_spec(source_path)
        frames = []
        for scen in ("onBody", "offBody"):
            if scen not in spec and "unified" not in spec:
                continue
            df_s = self.load_and_preprocess(
                source_path=source_path,
                scenario_filter=scen if scen in spec else None,
                preprocess_iq=preprocess_iq,
            )
            if df_s.empty:
                continue
            if device_ids and "dvc" in df_s.columns:
                df_s = df_s[df_s["dvc"].isin(device_ids)].copy()
            if max_per_scenario and len(df_s) > max_per_scenario:
                df_s = df_s.head(max_per_scenario).copy()
            frames.append(df_s)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=0, ignore_index=True)
        out = out.sort_values("global_frame_id").reset_index(drop=True)
        return out

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "UnifiedDataManager",
]