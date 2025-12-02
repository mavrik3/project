import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Sequence, Any, Callable
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    Low-level signal processing operations for feature extraction.
    Contains standalone methods for extracting specific signal properties.
    """
    
    @staticmethod
    def percentiles(v: np.ndarray, qs=(10, 25, 50, 75, 90)) -> List[float]:
        """Calculate percentiles of a signal."""
        if v.size == 0:
            return [0.0] * len(qs)
        return [float(np.percentile(v, q)) for q in qs]
    
    @staticmethod
    def extended_stats(vec: np.ndarray) -> List[float]:
        """Calculate extended statistics of a signal."""
        if vec.size == 0:
            return [0.0] * 6
        m = float(np.mean(vec))
        s = float(np.std(vec))
        mn = float(np.min(vec))
        mx = float(np.max(vec))
        p10, p90 = np.percentile(vec, [10, 90]) if vec.size else (0.0, 0.0)
        spread = float(p90 - p10)
        cv = float(s / (abs(m) + 1e-9))
        return [m, s, mn, mx, spread, cv]
    
    @staticmethod
    def spectral_bands(x: np.ndarray, nseg_max: int = 256, bands: int = 6) -> List[float]:
        """Extract spectral band energy features using Welch's method."""
        try:
            from scipy.signal import welch
            nper_desired = min(nseg_max, max(16, x.shape[0] // 4))
            nper = min(nper_desired, x.shape[0])
            if nper < 16:
                return [0.0] * bands
            _, Pxx = welch(x, nperseg=nper)
            if Pxx.size < bands:
                return list(np.pad(Pxx, (0, bands - Pxx.size))[:bands].astype(float))
            splits = np.array_split(Pxx, bands)
            return [float(np.sum(s)) for s in splits]
        except Exception:
            return [0.0] * bands
    
    @staticmethod
    def spectral_summary(x: np.ndarray) -> List[float]:
        """Extract spectral summary features (centroid, bandwidth, rolloff, flatness, crest)."""
        try:
            from scipy.signal import welch
            nper_desired = min(512, max(32, x.shape[0] // 4))
            nper = min(nper_desired, x.shape[0])
            if nper < 8:
                return [0.0] * 5
            f, Pxx = welch(x, nperseg=nper)
            Pxx = np.maximum(Pxx, 1e-12)
            ps = Pxx / np.sum(Pxx)
            centroid = float(np.sum(f * ps))
            bw = float(np.sqrt(np.sum(ps * (f - centroid) ** 2)))
            cdf = np.cumsum(ps)
            roll_idx = int(np.searchsorted(cdf, 0.9))
            rolloff = float(f[min(roll_idx, len(f) - 1)])
            geo = float(np.exp(np.mean(np.log(Pxx))))
            arith = float(np.mean(Pxx))
            flatness = float(geo / (arith + 1e-12))
            crest = float(np.max(Pxx) / (arith + 1e-12))
            return [centroid, bw, rolloff, flatness, crest]
        except Exception:
            return [0.0] * 5
    
    @staticmethod
    def autocorr_features(x: np.ndarray, max_lag: int = 32) -> List[float]:
        """Extract autocorrelation features from a signal."""
        if x.size < 4:
            return [0.0] * 6
        x = x - np.mean(x)
        if np.allclose(x.std(), 0):
            return [0.0] * 6
        ac = np.correlate(x, x, mode="full")
        ac = ac[ac.size // 2:]
        ac = ac / (ac[0] + 1e-12)
        l = min(max_lag, len(ac) - 1)
        segment = ac[1:l + 1]
        peak_lag = int(np.argmax(segment) + 1)
        peak_val = float(np.max(segment))
        mean_ac = float(np.mean(segment))
        std_ac = float(np.std(segment))
        # rough decay: lag where ac < 1/e
        below = np.where(ac < 1 / np.e)[0]
        decay = float(below[0]) if below.size else float(l)
        energy = float(np.sum(segment ** 2))
        return [peak_lag, peak_val, mean_ac, std_ac, decay, energy]
    
    @staticmethod
    def phase_autocorr_features(phase: np.ndarray, max_lag: int = 48) -> List[float]:
        """Extract autocorrelation features from phase signal."""
        if phase.size < 4:
            return [0.0] * 4
        # Center phase to remove CFO trend before autocorrelation
        phase_centered = phase - np.mean(phase)
        ac = np.correlate(phase_centered, phase_centered, mode="full")
        ac = ac[ac.size // 2:]
        ac = ac / (ac[0] + 1e-12)
        l = min(max_lag, len(ac) - 1)
        segment = ac[1:l + 1]
        if segment.size == 0:
            return [0.0] * 4
        peak_lag = int(np.argmax(segment) + 1)
        peak_val = float(np.max(segment))
        mean_ac = float(np.mean(segment))
        std_ac = float(np.std(segment))
        return [peak_lag, peak_val, mean_ac, std_ac]
    
    @staticmethod
    def iq_advanced_features(I: np.ndarray, Q: np.ndarray) -> List[float]:
        """Extract advanced features from I/Q signal pair."""
        if I.size == 0 or Q.size == 0:
            return [0.0] * 16  # Total features: 7 + 5 + 4 = 16
        
        # Correlation and power imbalance
        rho = float(np.corrcoef(I, Q)[0, 1]) if np.std(I) > 0 and np.std(Q) > 0 else 0.0
        pI = float(np.mean(I ** 2))
        pQ = float(np.mean(Q ** 2))
        imb = float((pI - pQ) / (pI + pQ + 1e-12))
        
        # Circularity coefficient
        cpx = I + 1j * Q
        E_x2 = np.mean(cpx ** 2)
        E_abs2 = np.mean(np.abs(cpx) ** 2) + 1e-12
        circ = float(np.abs(E_x2 / E_abs2))
        
        # Phase dynamics
        phase = np.unwrap(np.angle(cpx))
        dphi = np.diff(phase) if phase.size > 1 else np.array([0.0])
        dphi_mean = float(np.mean(dphi))
        dphi_std = float(np.std(dphi))
        dphi_pos_ratio = float(np.mean(dphi > 0))
        
        # PAPR on magnitude
        mag2 = np.abs(cpx) ** 2
        papr = float(np.max(mag2) / (np.mean(mag2) + 1e-12))
        
        # Spectral summaries on magnitude
        mag = np.sqrt(mag2 + 1e-12)
        spec = SignalProcessor.spectral_summary(mag)
        
        # Add phase autocorrelation features
        phase_acf = SignalProcessor.phase_autocorr_features(phase)
        
        return [rho, imb, circ, dphi_mean, dphi_std, dphi_pos_ratio, papr] + spec + phase_acf
    
    @staticmethod
    def short_time_dynamics(x: np.ndarray, segments: int = 8) -> List[float]:
        """Extract short-time dynamics features from a signal."""
        if x.size < segments:
            return [0.0] * 6
        splits = np.array_split(x, segments)
        means = np.array([np.mean(s) for s in splits])
        vars_ = np.array([np.var(s) for s in splits])
        return [
            float(means.mean()), float(means.std()),
            float(vars_.mean()), float(vars_.std()),
            float(np.max(means) - np.min(means)),
            float(np.max(vars_) - np.min(vars_))
        ]
    
    @staticmethod
    def to_matrix(x: np.ndarray) -> np.ndarray:
        """Convert signal to matrix representation."""
        if np.iscomplexobj(x):
            return np.stack([x.real, x.imag], axis=-1).astype(np.float32)
        if x.ndim == 1:
            return x.reshape(-1, 1).astype(np.float32)
        return x.astype(np.float32)
    
    @staticmethod
    def meta_vector(row: pd.Series) -> np.ndarray:
        """Extract metadata features from DataFrame row."""
        def f(v, d=0.0):
            try:
                return float(v) if pd.notna(v) else d
            except Exception:
                return d
        
        cfo = f(row.get("cfo_hz"))
        rssi = f(row.get("rssi_db"))
        length = f(row.get("num_samples") if "num_samples" in row else row.get("len"))
        Llog = np.log1p(max(0.0, length))
        
        # Channel encoding (either use actual channel or set to zero)
        ch_sin, ch_cos = 0.0, 0.0
        if "channel" in row and pd.notna(row["channel"]):
            try:
                ch = float(row["channel"])
                ch_sin = np.sin(2 * np.pi * ch / 40)  # Assuming 40 channels (BLE)
                ch_cos = np.cos(2 * np.pi * ch / 40)
            except:
                pass
                
        rx = f(row.get("rx_idx"))
        return np.array([cfo, rssi, Llog, ch_sin, ch_cos, rx], dtype=np.float32)


class FeatureExtractor:
    """
    High-level feature extraction from signal data.
    Combines multiple low-level features into feature vectors with named dimensions.
    """
    
    # Constants for feature naming
    META_NAMES = ["meta_cfo_hz", "meta_rssi_db", "meta_log_len", 
                  "meta_channel_sin", "meta_channel_cos", "meta_rx_idx"]
    
    @staticmethod
    def _signal_feature_names(C: int) -> List[str]:
        """Generate feature names for signal features based on channel count."""
        names: List[str] = []
        names.append("len_T")
        
        for c in range(C):
            names += [f"ch{c}_mean", f"ch{c}_std", f"ch{c}_min", f"ch{c}_max", 
                      f"ch{c}_spread_p90_p10", f"ch{c}_cv"]
            names += [f"ch{c}_p{q}" for q in (10, 25, 50, 75, 90)]
        
        if C > 1:
            # upper triangle covariance (row-major: (0,0),(0,1),(1,1),...)
            for i in range(C):
                for j in range(i, C):
                    names.append(f"cov_ch{i}{j}")
        else:
            names.append("cov_placeholder")
            
        if C == 2:
            names += ["mag_mean", "mag_std", "mag_min", "mag_max", "mag_spread_p90_p10", "mag_cv"]
            names += [f"mag_p{q}" for q in (10, 25, 50, 75, 90)]
            names += ["dphi_mean_abs", "dphi_std"]
            names += [f"mag_band{k}" for k in range(6)]
        else:
            names += ["sig_mean", "sig_std", "sig_min", "sig_max", "sig_spread_p90_p10", "sig_cv"]
            names += [f"sig_p{q}" for q in (10, 25, 50, 75, 90)]
            names += ["dphi_mean_abs_placeholder", "dphi_std_placeholder"]
            names += [f"sig_band{k}" for k in range(6)]
            
        return names
    
    @staticmethod
    def _advanced_feature_names(C: int) -> List[str]:
        """Generate feature names for advanced features based on channel count."""
        names: List[str] = []
        
        # Autocorrelation on magnitude (or single channel if C!=2)
        names += ["acf_peak_lag", "acf_peak_val", "acf_mean", "acf_std", "acf_decay", "acf_energy"]
        
        # Short-time dynamics on magnitude/signal
        names += ["stdyn_mean_mean", "stdyn_mean_std", 
                  "stdyn_var_mean", "stdyn_var_std", 
                  "stdyn_mean_range", "stdyn_var_range"]
                  
        # Spectral summary on magnitude/signal
        names += ["spec_centroid", "spec_bandwidth", "spec_rolloff90", 
                  "spec_flatness", "spec_crest"]
                  
        if C == 2:
            names += ["iq_corr", "iq_power_imbalance", "iq_circularity", 
                      "dphi_mean", "dphi_std", "dphi_pos_ratio", "papr",
                      "mag_spec_centroid", "mag_spec_bandwidth", "mag_spec_rolloff90", 
                      "mag_spec_flatness", "mag_spec_crest",
                      "phase_acf_peak_lag", "phase_acf_peak_val", "phase_acf_mean", "phase_acf_std"]
                      
        return names
    
    @staticmethod
    def compute_frame_features(x_mat: np.ndarray, advanced: bool = True) -> np.ndarray:
        """
        Compute comprehensive feature vector from signal matrix.
        
        Args:
            x_mat: Signal matrix of shape (T, C) where T is time and C is channels
            advanced: Whether to compute advanced features (more expensive)
            
        Returns:
            Feature vector as numpy array
        """
        if x_mat.size == 0:
            return np.zeros(64, dtype=np.float32)
            
        T, C = x_mat.shape
        feats: List[float] = [float(T)]
        
        # Basic statistics for each channel
        for c in range(C):
            col = x_mat[:, c]
            feats += SignalProcessor.extended_stats(col)
            feats += SignalProcessor.percentiles(col)
            
        # Covariance features
        if C > 1:
            cov = np.cov(x_mat.T)
            feats += list(cov[np.triu_indices(C)])
        else:
            feats += [0.0]
            
        # IQ-specific features when 2 channels
        if C == 2:
            I, Q = x_mat[:, 0], x_mat[:, 1]
            cpx = I + 1j * Q
            mag = np.abs(cpx)
            phase = np.unwrap(np.angle(cpx))
            dphi = np.diff(phase) if phase.size > 1 else np.array([0.0])
            
            feats += SignalProcessor.extended_stats(mag)
            feats += SignalProcessor.percentiles(mag)
            feats += [float(np.mean(np.abs(dphi))), float(np.std(dphi))]
            feats += SignalProcessor.spectral_bands(mag)
            
            if advanced:
                feats += SignalProcessor.autocorr_features(mag)
                feats += SignalProcessor.short_time_dynamics(mag)
                feats += SignalProcessor.spectral_summary(mag)
                feats += SignalProcessor.iq_advanced_features(I, Q)
        else:
            # Single channel features
            v = x_mat[:, 0]
            feats += SignalProcessor.extended_stats(v) + SignalProcessor.percentiles(v)
            feats += [0.0, 0.0]  # placeholders for dphi stats
            feats += SignalProcessor.spectral_bands(v)
            
            if advanced:
                feats += SignalProcessor.autocorr_features(v)
                feats += SignalProcessor.short_time_dynamics(v)
                feats += SignalProcessor.spectral_summary(v)
                
        # Replace NaNs and infinities with zeros
        return np.nan_to_num(np.array(feats, dtype=np.float32), 
                             nan=0.0, posinf=0.0, neginf=0.0)
    
    @staticmethod
    def get_feature_names(n_channels: int = 2, advanced: bool = True, include_meta: bool = True) -> List[str]:
        """
        Get complete list of feature names based on configuration.
        
        Args:
            n_channels: Number of signal channels (1 or 2)
            advanced: Whether advanced features are included
            include_meta: Whether metadata features are included
            
        Returns:
            List of feature names in order
        """
        names = FeatureExtractor._signal_feature_names(n_channels)
        
        if advanced:
            names += FeatureExtractor._advanced_feature_names(n_channels)
            
        if include_meta:
            names += FeatureExtractor.META_NAMES
            
        return names


class FeatureEngineering:
    """
    Unified feature engineering for BAN Auth Project.
    Extracts features from preprocessed dataframes for machine learning.
    """
    
    def __init__(self):
        """Initialize feature engineering class."""
        pass
    
    def extract_features(self,
                        df: pd.DataFrame,
                        label_col: str,
                        max_per_class: int,
                        include_meta: bool = True,
                        advanced_features: bool = True,
                        group_by: Optional[str] = None,
                        balance_by: Optional[List[str]] = None,
                        remove_device_fingerprint: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
        """
        Extract features from a preprocessed DataFrame.
        
        Args:
            df: Preprocessed DataFrame with IQ data
            label_col: Column name for class labels
            max_per_class: Maximum samples per class
            include_meta: Whether to include metadata features
            advanced_features: Whether to compute advanced (more expensive) features
            group_by: Optional column name to use for grouping (e.g., 'session'). Defaults to 'dvc'.
            balance_by: Optional list of columns to balance within each class (e.g., ['pos_label','session']).
                When provided, sampling within each class is balanced across the Cartesian combinations of these columns.
                Columns not present in df are ignored.
            remove_device_fingerprint: If True, apply per-group IQ normalization (amplitude RMS) prior to feature engineering.
                This reduces device-specific fingerprints before any linear/nonlinear feature transforms are computed.
            
        Returns:
            Tuple of (X, y, groups, kept_indices, feature_names)
                X: Feature matrix of shape (n_samples, n_features)
                y: Class labels
                groups: Group IDs for grouped cross-validation
                kept_indices: Indices of samples kept from original DataFrame
                feature_names: Names of features in order
        """
        if df.empty or label_col not in df.columns:
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([]), [], []
        
        # Balance classes by sampling
        parts = []
        rng = np.random.default_rng(42)
        # Determine balancing keys present in the dataframe
        bal_keys: List[str] = []
        if balance_by:
            bal_keys = [k for k in balance_by if k != label_col and k in df.columns]
        
        for lab, sub in df.groupby(label_col):
            if pd.isna(lab) or sub.empty:
                continue
            if bal_keys:
                # Balance across combinations of bal_keys within this class
                combos = list(sub.groupby(bal_keys))
                if len(combos) == 0:
                    # Fallback to simple per-class sampling
                    if len(sub) > max_per_class:
                        parts.append(sub.sample(max_per_class, random_state=int(rng.integers(1 << 30))))
                    else:
                        parts.append(sub)
                else:
                    # Target per-combo cap to distribute fairly across combos
                    per_combo_cap = max(1, int(np.floor(max_per_class / max(1, len(combos)))))
                    for _, combo_df in combos:
                        if combo_df.empty:
                            continue
                        if len(combo_df) > per_combo_cap:
                            parts.append(combo_df.sample(per_combo_cap, random_state=int(rng.integers(1 << 30))))
                        else:
                            parts.append(combo_df)
            else:
                # Original behavior: per-class cap
                if len(sub) > max_per_class:
                    parts.append(sub.sample(max_per_class, random_state=int(rng.integers(1 << 30))))
                else:
                    parts.append(sub)
                
        if not parts:
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([]), [], []
            
        work = pd.concat(parts, ignore_index=True)
        
        # -----------------------------
        # Optional: Pre-feature fingerprint removal on IQ
        # -----------------------------
        grp_key = group_by if group_by else "dvc"
        iq_scale_by_group: Dict[Any, float] = {}
        if remove_device_fingerprint and grp_key in work.columns:
            # Compute a robust per-group amplitude RMS using a quick pass
            per_group_rms: Dict[Any, List[float]] = {}
            for _, r in tqdm(work.iterrows(), total=len(work), desc=f"Calibrating per-{grp_key} RMS"):
                xin = r.get("frame_inline", None)
                if xin is None:
                    xin = r.get("frame", None)
                if xin is None:
                    xin = r.get("iq", None)
                if xin is None:
                    xin = r.get("processed_frame", None)
                if xin is None:
                    continue
                x = np.asarray(xin)
                if x.ndim == 2 and x.shape[1] == 2 and not np.iscomplexobj(x):
                    x = x[:, 0].astype(np.float32) + 1j * x[:, 1].astype(np.float32)
                if not np.iscomplexobj(x):
                    continue
                x_al = self.preprocess_signal(x)
                if x_al is None or x_al.size < 16:
                    continue
                # amplitude RMS of aligned signal
                rms = float(np.sqrt(np.mean(np.abs(x_al) ** 2)))
                if not np.isfinite(rms) or rms <= 1e-6:
                    continue
                gval = r.get(grp_key)
                per_group_rms.setdefault(gval, []).append(rms)
            # Aggregate (median) and store normalization factors
            for gval, vals in per_group_rms.items():
                if len(vals) == 0:
                    continue
                s = float(np.median(vals))
                iq_scale_by_group[gval] = s if s > 1e-6 else 1.0
            logger.info(f"Applied pre-feature fingerprint removal using per-{grp_key} RMS (groups={len(iq_scale_by_group)})")
        
        # Track results
        feats = []
        labels = []
        groups = []
        kept_idx = []
        feat_names: List[str] = []
        first_done = False
        
        # Process each row
        for _, r in tqdm(work.iterrows(), total=len(work), desc=f"Extracting {label_col} features"):
            # Find IQ data
            xin = r.get("frame_inline", None)
            if xin is None:
                xin = r.get("frame", None)
            if xin is None:
                xin = r.get("iq", None)
            if xin is None:
                xin = r.get("processed_frame", None)
            if xin is None:
                continue
                
            # Convert to numpy array if needed
            x = np.asarray(xin)
            
            # Handle 2D array with I/Q channels
            if x.ndim == 2 and x.shape[1] == 2 and not np.iscomplexobj(x):
                x = x[:, 0].astype(np.float32) + 1j * x[:, 1].astype(np.float32)
                
            # Skip if not complex
            if not np.iscomplexobj(x):
                continue
                
            # Get label and normalize it
            lbl = r.get(label_col)
            if pd.isna(lbl):
                continue
                
            if isinstance(lbl, (int, np.integer)):
                lbl_norm = f"{label_col}_{int(lbl)}"
            elif isinstance(lbl, (float, np.floating)):
                lbl_norm = f"{label_col}_{int(lbl)}" if float(lbl).is_integer() else f"{label_col}_{str(lbl)}"
            else:
                lbl_norm = f"{label_col}_{str(lbl)}"
                
            # Preprocess the signal: normalize and convert to matrix
            x_aligned = self.preprocess_signal(x)
            if x_aligned is None or x_aligned.size < 16:
                continue
            
            # Apply pre-feature fingerprint removal (amplitude normalization) if available
            if remove_device_fingerprint and iq_scale_by_group:
                gval = r.get(grp_key)
                s = iq_scale_by_group.get(gval, None)
                if s is not None and s > 1e-6:
                    x_aligned = x_aligned / s
            
            # Convert to matrix and extract features
            mat = SignalProcessor.to_matrix(x_aligned)
            f_sig = FeatureExtractor.compute_frame_features(mat, advanced=advanced_features)
            
            # Generate feature names on first successful extraction
            if not first_done:
                names = FeatureExtractor._signal_feature_names(mat.shape[1])
                if advanced_features:
                    names += FeatureExtractor._advanced_feature_names(mat.shape[1])
                feat_names = names.copy()
                first_done = True
                
            # Add metadata features if requested
            if include_meta:
                f_sig = np.concatenate([f_sig, SignalProcessor.meta_vector(r)], axis=0)
                
            # Store results
            feats.append(f_sig)
            labels.append(lbl_norm)
            # Group selection
            grp_key_eff = group_by if group_by else "dvc"
            groups.append(r.get(grp_key_eff))
            kept_idx.append(int(r.get("frame_idx", -1)))
            
        # Check for valid results
        if not feats or len(set(labels)) < 2:
            return np.zeros((0, 0), dtype=np.float32), np.array([]), np.array([]), [], []
            
        # Build output arrays
        X = np.vstack(feats).astype(np.float32)
        if include_meta and feat_names:
            feat_names = feat_names + FeatureExtractor.META_NAMES
            
        # y as prefix-normalized strings for stability
        y = np.array(labels, dtype=object)
        g = np.array(groups, dtype=object)
        
        return X, y, g, kept_idx, feat_names

    def preprocess_signal(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess a signal for feature extraction.
        
        Args:
            x: Complex IQ signal
            
        Returns:
            Preprocessed signal or None if preprocessing fails
        """
        from data_manager import DataPreprocessor
        
        # Try to extract preamble
        x_aligned = DataPreprocessor.extract_preamble(x, candidate_sps=[95, 100, 105]) # For 100 MSps recording of 1 Mbps BLE signal
        # If preamble extraction fails, fallback to simple transient trimming
        if x_aligned is None:
            x_aligned = DataPreprocessor.trim_transient(x, drop_min=50, frac=0.05)
            
        # Skip if still invalid
        if x_aligned is None or x_aligned.size < 16:
            return None
            
        # Normalize the signal
        x_aligned = DataPreprocessor.normalize_iq_signal(x_aligned)
        
        return x_aligned
    
    def extract_features_for_training(self,
                                     df: pd.DataFrame,
                                     label_col: str,
                                     max_per_class: int = 5000,
                                     include_meta: bool = True,
                                     advanced_features: bool = True,
                                     remove_device_fingerprint: bool = False,
                                     group_by: Optional[str] = None,
                                     balance_by: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract features and prepare data for training.
        
        Args:
            df: Preprocessed DataFrame with IQ data
            label_col: Column name for class labels
            max_per_class: Maximum samples per class
            include_meta: Whether to include metadata features
            advanced_features: Whether to compute advanced features
            remove_device_fingerprint: Whether to remove device fingerprints from features (applied pre-feature on IQ)
            group_by: Optional column name to use for grouping (e.g., 'session'). Defaults to 'dvc'.
            balance_by: Optional list of columns to balance within each class (e.g., ['pos_label','session']).
            
        Returns:
            Dictionary with keys:
                'X': Feature matrix
                'y': Class labels
                'groups': Group IDs
                'feature_names': Feature names
                'class_counts': Count of samples per class
                'aux_labels': Optional dict of aligned auxiliary labels (e.g., dvc, pos_label, session)
        """
        # Extract features (now supports pre-feature fingerprint removal)
        X, y, groups, kept_idx, feature_names = self.extract_features(
            df, label_col, max_per_class, include_meta, advanced_features, group_by=group_by, balance_by=balance_by,
            remove_device_fingerprint=remove_device_fingerprint
        )
        
        # Check if extraction succeeded
        if X.size == 0 or len(np.unique(y)) < 2:
            logger.warning(f"Feature extraction failed for {label_col}. "
                          f"Got X shape: {X.shape}, unique classes: {len(np.unique(y)) if y.size > 0 else 0}")
            return {
                'X': np.zeros((0, 0), dtype=np.float32),
                'y': np.array([]),
                'groups': np.array([]),
                'feature_names': [],
                'class_counts': {},
                'aux_labels': {}
            }
        
        # Removed post-feature fingerprint removal to ensure integrity of engineered features.
        # Device fingerprint mitigation is now applied to IQ prior to feature computation.
        
        # Count samples per class
        unique_classes, counts = np.unique(y, return_counts=True)
        class_counts = {str(cls): int(count) for cls, count in zip(unique_classes, counts)}
        
        logger.info(f"Extracted {X.shape[0]} samples with {X.shape[1]} features "
                   f"for {len(unique_classes)} classes of {label_col}")
        
        # Build auxiliary labels aligned with the kept order using frame_idx mapping if available
        aux_labels: Dict[str, np.ndarray] = {}
        try:
            if kept_idx and 'frame_idx' in df.columns:
                # Build a lookup from frame_idx to values for desired columns
                cols = [c for c in ['dvc', 'pos_label', 'session'] if c in df.columns]
                if cols:
                    sub = df[['frame_idx'] + cols].dropna(subset=['frame_idx']).copy()
                    sub['frame_idx'] = sub['frame_idx'].astype(int)
                    lut = {int(r['frame_idx']): {k: r[k] for k in cols} for _, r in sub.iterrows()}
                    # Reconstruct aligned arrays
                    aligned = {k: [] for k in cols}
                    for fi in kept_idx:
                        d = lut.get(int(fi), {})
                        for k in cols:
                            aligned[k].append(d.get(k, None))
                    for k, v in aligned.items():
                        aux_labels[k] = np.array(v, dtype=object)
        except Exception:
            pass
        
        return {
            'X': X,
            'y': y,
            'groups': groups,
            'feature_names': feature_names,
            'class_counts': class_counts,
            'aux_labels': aux_labels,
            'kept_frame_indices': kept_idx,
        }
    
    def extract_all_features(self, 
                           df: pd.DataFrame,
                           task_config: Dict[str, Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Extract features for multiple classification tasks from a single DataFrame.
        
        Args:
            df: Preprocessed DataFrame with IQ data
            task_config: Configuration for each task with structure:
                {
                    "task_name": {
                        "label_col": str,
                        "max_per_class": int,
                        "include_meta": bool,
                        "advanced_features": bool,
                        "remove_fingerprint": bool
                    }
                }
            
        Returns:
            Dictionary mapping task names to feature extraction results
        """
        if df.empty:
            return {}
            
        # Default configuration
        default_config = {
            "max_per_class": 5000,
            "include_meta": True,
            "advanced_features": True,
            "remove_fingerprint": False
        }
        
        # Use default tasks if none specified
        if task_config is None:
            task_config = {}
            
            # Add scenario classification if column exists
            if "scenario" in df.columns and len(df["scenario"].dropna().unique()) > 1:
                task_config["scenario"] = {"label_col": "scenario"}
                
            # Add device classification if column exists
            if "dvc" in df.columns and len(df["dvc"].dropna().unique()) > 1:
                task_config["device"] = {"label_col": "dvc"}
                
            # Add position classification if column exists
            if "pos_label" in df.columns and len(df["pos_label"].dropna().unique()) > 1:
                task_config["position"] = {"label_col": "pos_label", "remove_fingerprint": True}
                
            # Add movement classification if column exists
            if "movement" in df.columns and len(df["movement"].dropna().unique()) > 1:
                task_config["movement"] = {"label_col": "movement"}
        
        results = {}
        
        # Process each task
        for task_name, config in task_config.items():
            task_cfg = {**default_config, **config}
            
            logger.info(f"Extracting features for task: {task_name}")
            
            # Create task-specific DataFrame if needed
            task_df = df
            if "scenario_filter" in task_cfg and task_cfg["scenario_filter"] is not None:
                if "scenario" in df.columns:
                    task_df = df[df["scenario"] == task_cfg["scenario_filter"]].copy()
                    if task_df.empty:
                        logger.warning(f"No data for scenario filter: {task_cfg['scenario_filter']}")
                        continue
            
            # Extract features
            result = self.extract_features_for_training(
                task_df,
                label_col=task_cfg["label_col"],
                max_per_class=task_cfg["max_per_class"],
                include_meta=task_cfg["include_meta"],
                advanced_features=task_cfg["advanced_features"],
                remove_device_fingerprint=task_cfg.get("remove_fingerprint", False)
            )
            
            if result["X"].size > 0:
                results[task_name] = result
            
        return results