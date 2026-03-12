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
        """List dataset directories available under a root path.

        A directory is considered a valid dataset if it contains either an
        ``index.parquet`` or ``index.csv`` file.

        Args:
            root: Root directory to search for datasets.

        Returns:
            Sorted list of Paths to valid dataset directories.
        """
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
        """Load a dataset index and its associated dataset card from disk.

        Reads ``index.parquet`` (preferred) or ``index.csv`` from the given
        directory, and ``card.json`` if present.

        Args:
            dir_path: Path to the dataset directory.

        Returns:
            Tuple of (DataFrame, card dict). The DataFrame contains the dataset
            index rows; the dict contains metadata from ``card.json`` (empty
            dict if the card is absent or unreadable).
        """
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
        """Load a DataFrame from a pickle file.

        Args:
            pkl_path: Path to the ``.pkl`` or ``.pickle`` file.
            preproc_path: Unused; reserved for future preprocessed-cache
                support.
            force_recompute: Unused; reserved for future cache-bypass support.
            debug_toy_subset: When True, randomly samples 2,000 rows from the
                loaded DataFrame to speed up development iterations.

        Returns:
            The loaded DataFrame with the index reset, or an empty DataFrame
            if the file does not exist or cannot be loaded.
        """
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
        """Load off-body rows from a unified pickle file.

        Reads the unified pickle and filters rows where the scenario column
        equals ``"offbody"`` (case-insensitive). Useful as a fallback when a
        dedicated off-body dataset file is not available.

        Args:
            unified_pkl_path: Path to the unified ``.pkl`` file.
            max_rows: If provided, truncates the result to this many rows.

        Returns:
            DataFrame containing only off-body rows, with the index reset.
            Returns an empty DataFrame if the file cannot be loaded.
        """
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
        """Find the first matching column name from a list of candidates.

        Performs an exact match first, then a case-insensitive fuzzy match
        against the DataFrame's column names.

        Args:
            df: DataFrame whose columns are searched.
            candidates: Ordered list of preferred column name strings.

        Returns:
            The matching column name as it appears in the DataFrame, or None
            if no candidate matches.
        """
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
        """Rename dataset columns to the project's canonical names.

        Maps common alternative column names for channel, scenario, device, and
        position to their canonical equivalents (``channel``, ``scenario``,
        ``dvc``, ``pos_label``). Also ensures the channel column is cast to
        ``int`` and adds a ``global_frame_id`` column if one is absent.

        Args:
            df: Input DataFrame with potentially non-canonical column names.

        Returns:
            A copy of the DataFrame with standardized column names.
        """
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
        """Identify the IQ signal column in a DataFrame.

        Searches for known IQ column name candidates and checks that the
        column contains object-dtype entries (i.e., array-like IQ samples).

        Args:
            df: DataFrame to inspect.

        Returns:
            Name of the detected IQ column, or None if none is found.
        """
        for c in IQ_CANDIDATES:
            if c in df.columns:
                # quick heuristic: object or list-like entries
                if df[c].dtype == object:
                    return c
        return None

    @staticmethod
    def is_valid_iq(arr) -> bool:
        """Check whether an array-like value looks like a valid IQ signal.

        A value is considered valid if it is a list, tuple, or NumPy array
        with more than four elements.

        Args:
            arr: Value to check.

        Returns:
            True if ``arr`` appears to be a non-trivial IQ signal, False
            otherwise.
        """
        if arr is None:
            return False
        if isinstance(arr, (list, tuple)) and len(arr) > 4:
            return True
        if isinstance(arr, np.ndarray) and arr.ndim <= 2 and arr.size > 4:
            return True
        return False

    @staticmethod
    def to_complex(arr) -> Optional[np.ndarray]:
        """Convert an array-like IQ representation to a complex64 NumPy array.

        Handles the following input formats:

        - Already-complex NumPy arrays.
        - Real-valued 2-D arrays of shape ``(N, 2)`` interpreted as
          ``(I, Q)`` pairs.
        - Lists or tuples in the same formats as above.

        Args:
            arr: Input IQ data in any supported format.

        Returns:
            Complex64 NumPy array, or None if the input cannot be interpreted
            as a complex signal.
        """
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
        """Extract the preamble portion of a BLE IQ frame.

        Currently uses a placeholder heuristic that returns the first
        ``preamble_len_bits * 2`` samples when the frame is long enough.
        Future versions will implement proper preamble detection.

        Args:
            x: Complex IQ signal array.
            preamble_len_bits: Expected length of the preamble in bits.
            candidate_sps: Candidate samples-per-symbol values (reserved for
                future use).
            min_score: Minimum detection confidence threshold (reserved for
                future use).

        Returns:
            NumPy array containing the extracted preamble samples, or None if
            the frame is too short.
        """
        if x.size < preamble_len_bits * 2:
            return None
        return x[: preamble_len_bits * 2]

    @staticmethod
    def normalize_iq_signal(x: np.ndarray) -> np.ndarray:
        """Zero-mean and unit-variance normalize a complex IQ signal.

        Subtracts the mean and divides by the standard deviation. If the
        standard deviation is zero (constant signal), only mean subtraction is
        applied.

        Args:
            x: Input complex or real IQ signal array.

        Returns:
            Normalized signal array of the same shape and dtype as ``x``.
        """
        if x.size == 0:
            return x
        xr = x - np.mean(x)
        std = np.std(xr)
        if std > 0:
            xr = xr / std
        return xr

    @staticmethod
    def trim_transient(x: np.ndarray, drop_min: int = 800, frac: float = 0.08) -> np.ndarray:
        """Remove transient samples from the beginning of an IQ frame.

        Drops an initial segment whose length is the smaller of
        ``len(x) * frac`` and ``drop_min``. Frames shorter than ``drop_min``
        are returned unchanged.

        Args:
            x: Input IQ signal array.
            drop_min: Minimum frame length required before any trimming is
                applied; also the upper bound on the number of samples dropped.
            frac: Fraction of the frame length to drop, capped at ``drop_min``.

        Returns:
            Trimmed signal array.
        """
        if x.size <= drop_min:
            return x
        k = int(min(len(x) * frac, drop_min))
        return x[k:]

    @staticmethod
    def preprocess_iq_frames(df: pd.DataFrame, target_length: int = 2000) -> pd.DataFrame:
        """Preprocess raw IQ frames in a DataFrame and store the results.

        For each row, converts the raw IQ entry to a complex array, applies
        :meth:`normalize_iq_signal` and :meth:`trim_transient`, then truncates
        to ``target_length`` samples. The processed signals are stored in a new
        ``iq_proc`` column.

        Args:
            df: DataFrame containing a raw IQ column detected by
                :meth:`detect_iq_column`.
            target_length: Maximum number of samples to keep per frame after
                trimming. Frames shorter than this are kept as-is.

        Returns:
            A copy of the input DataFrame with an additional ``iq_proc`` column
            holding the processed complex arrays (or None for invalid frames).
            Returns the original DataFrame unchanged if no IQ column is found.
        """
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
        """Filter a DataFrame to off-body rows and optionally limit its size.

        Args:
            df: Input DataFrame, expected to have a ``scenario`` column.
            max_rows: If provided, the result is truncated to this many rows.

        Returns:
            DataFrame containing only rows where ``scenario == "offbody"``
            (case-insensitive), with the index reset. Returns the original
            DataFrame unchanged if it is empty.
        """
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
        """Filter a DataFrame to on-body rows and optionally limit its size.

        Args:
            df: Input DataFrame, expected to have a ``scenario`` column.
            max_rows: If provided, the result is truncated to this many rows.

        Returns:
            DataFrame containing only rows where ``scenario == "onbody"``
            (case-insensitive), with the index reset. Returns the original
            DataFrame unchanged if it is empty.
        """
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
        """Verify that all required data sources exist on disk.

        Resolves *source_path* to a spec dictionary and checks whether each
        referenced file or dataset directory is accessible.

        Args:
            scenario_filter: If ``"onBody"`` or ``"offBody"``, only the
                matching source (or the unified fallback) is checked.
                Otherwise, all resolved sources are verified.
            source_path: Data source specification. Accepts a file path,
                directory path, or a dict mapping scenario keys to paths.
                Falls back to ``self.root_dir`` if None.

        Returns:
            dict with keys:

            - ``ok`` (bool): True when all required sources are present.
            - ``spec`` (dict): Resolved source paths as strings.
            - ``exists`` (list[str]): Paths that exist.
            - ``missing`` (list[str]): Paths that are missing.
            - ``message`` (str): ``"OK"`` or a short error description.
        """
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
        """Load a dataset from disk, standardize its columns, and optionally preprocess IQ frames.

        Results are cached in ``self.data_cache`` keyed by source path,
        scenario, row limit, and IQ-preprocessing flag so subsequent calls
        with the same arguments are served from memory.

        Args:
            source_path: Path to the dataset. Accepts a pickle file, a
                dataset directory (containing ``index.parquet``/``index.csv``),
                or a dict mapping scenario keys to paths.
            preprocess_iq: If True, runs
                :meth:`DataPreprocessor.preprocess_iq_frames` on the loaded
                DataFrame to normalize and trim IQ signal frames.
            scenario_filter: Filter loaded rows to ``"onBody"`` or
                ``"offBody"`` after loading.
            max_rows: Limit the result to this many rows (head truncation).
            force_reload: If True, bypass the cache and reload from disk.

        Returns:
            DataFrame with standardized column names, filtered by scenario
            (if requested), sorted by ``global_frame_id``. Returns an empty
            DataFrame when no valid source is found.

        Raises:
            ValueError: If the loaded DataFrame has no ``channel`` column
                after standardization.
        """
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
        """Load and return rows for a specific scenario (on-body or off-body).

        Convenience wrapper around :meth:`load_and_preprocess` that sets
        *scenario_filter* automatically.

        Args:
            scenario: ``"onBody"`` or ``"offBody"``.
            source_path: Data source specification (file, directory, or dict).
            preprocess_iq: Whether to preprocess IQ frames after loading.
            max_rows: Maximum number of rows to return.

        Returns:
            DataFrame filtered to the requested scenario.
        """
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
        """Load and return rows for one or more specific device IDs.

        Args:
            device_id: A single device ID (int) or list of device IDs to
                filter on the ``dvc`` column.
            scenario: Optional scenario filter (``"onBody"`` or
                ``"offBody"``).
            source_path: Data source specification.
            preprocess_iq: Whether to preprocess IQ frames.
            max_rows: Maximum number of rows to return before device
                filtering.

        Returns:
            DataFrame containing only rows matching the requested device(s).
            Returns an empty DataFrame if the source is empty or has no
            ``dvc`` column.
        """
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
        """Load and return rows for one or more body positions.

        Args:
            position: A single position label (str) or list of labels to
                filter on the ``pos_label`` column.
            scenario: Optional scenario filter (``"onBody"`` or
                ``"offBody"``).
            source_path: Data source specification.
            preprocess_iq: Whether to preprocess IQ frames.
            max_rows: Maximum number of rows to return before position
                filtering.

        Returns:
            DataFrame containing only rows matching the requested position(s).
            Returns an empty DataFrame if the source is empty or has no
            ``pos_label`` column.
        """
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
        """Load and concatenate on-body and off-body data into a single DataFrame.

        Attempts to load both ``"onBody"`` and ``"offBody"`` scenarios from
        *source_path*, optionally filters by device IDs and row limits per
        scenario, then returns a combined DataFrame sorted by
        ``global_frame_id``.

        Args:
            source_path: Data source specification (file, directory, or dict).
            device_ids: Optional list of device IDs to retain. Rows with
                other device IDs are dropped.
            preprocess_iq: Whether to preprocess IQ frames after loading.
            max_per_scenario: Maximum number of rows to keep per scenario
                before concatenation.

        Returns:
            Combined DataFrame with rows from both scenarios, sorted by
            ``global_frame_id``. Returns an empty DataFrame if no scenario
            data could be loaded.
        """
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