import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence

import numpy as np
import pandas as pd
# Force headless matplotlib backend to avoid Tkinter issues on Windows
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
try:
    matplotlib.use("Agg", force=True)
except Exception:
    pass
import matplotlib.pyplot as plt

from data_manager import UnifiedDataManager, DataPreprocessor
from feature_engineering import FeatureEngineering
# Removed import of feature-space fingerprint removal to avoid confusion
# from ml_dl_framework import ModelType, TrainingConfig, remove_device_fingerprint
from training_api import train_classifier, classify_with_visuals

# Try to import joblib for model (de)serialization
try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class AnalysisFramework:
    """
    Unified analysis framework for BAN Authentication Project.
    Orchestrates all analyses from a single interface, ensuring consistent
    data flow and methodology across different authentication scenarios.
    """
    
    def __init__(
        self, 
        data_root: Union[str, Path],
        output_root: Optional[Union[str, Path]] = None,
        max_samples_per_class: int = 10000,
        accuracy_threshold: float = 0.90,
        use_advanced_features: bool = True,
        use_deep_learning: bool = True,
        default_balance_by: Optional[List[str]] = None,
    ):
        """
        Initialize the analysis framework.
        
        Args:
            data_root: Root directory containing datasets
            output_root: Root directory for saving results (default: data_root/results)
            max_samples_per_class: Maximum samples per class for balanced datasets
            accuracy_threshold: Accuracy threshold for model selection
            use_advanced_features: Whether to use advanced features for classification
            use_deep_learning: Whether to include deep learning models
            default_balance_by: Default list of columns to balance within each class (e.g., ['dvc','pos_label','session'])
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root) if output_root else self.data_root / "results"
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        self.max_samples_per_class = max_samples_per_class
        self.accuracy_threshold = accuracy_threshold
        self.use_advanced_features = use_advanced_features
        self.use_deep_learning = use_deep_learning
        self.default_balance_by = default_balance_by or []
        
        # Initialize components
        self.data_manager = UnifiedDataManager(root_dir=self.data_root)
        self.feature_engine = FeatureEngineering()
        
        # Analysis results cache
        self.results_cache = {}
        
        logger.info(f"Initialized Analysis Framework")
        logger.info(f"  Data root: {self.data_root}")
        logger.info(f"  Output root: {self.output_root}")
        logger.info(f"  Max samples per class: {self.max_samples_per_class}")
        logger.info(f"  Use advanced features: {self.use_advanced_features}")
        logger.info(f"  Use deep learning: {self.use_deep_learning}")
        if self.default_balance_by:
            logger.info(f"  Default balance_by: {self.default_balance_by}")
    
    def _get_output_dir(self, analysis_name: str) -> Path:
        """Get output directory for a specific analysis."""
        output_dir = self.output_root / analysis_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def run_all_analyses(self, source_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Run all major analyses and return an aggregated summary.
        Creates a combined JSON and CSV summary under the output root.
        """
        all_dir = self._get_output_dir("all_analyses")
        results: Dict[str, Any] = {}
        # Run analyses with graceful failure handling
        def _safe(run_fn, key: str, *args, **kwargs):
            try:
                res = run_fn(*args, **kwargs)
                results[key] = res
                return res
            except Exception as e:
                logger.error(f"{key} failed: {e}")
                results[key] = {"error": str(e)}
                return results[key]

        pos = _safe(self.analyze_offbody_position, "offbody_position", source_path)
        ver = _safe(self.analyze_offbody_verification, "offbody_verification", source_path)
        per_pos = _safe(self.analyze_offbody_device_id_per_position, "offbody_deviceid_per_position", source_path)
        dev_both = _safe(self.analyze_device_identification, "device_identification", "both", source_path)
        mov = _safe(self.analyze_onbody_movement, "onbody_movement", True, source_path)
        mix = _safe(self.analyze_mixed_scenario, "mixed_scenario", source_path)
        hier = _safe(self.analyze_hierarchical_classification, "hierarchical_classification", source_path)

        # Build summary dictionary
        summary: Dict[str, Any] = {}
        try:
            # OffBody position (global)
            gp = (pos or {}).get("global", {})
            summary["offbody_position_global_accuracy"] = (gp.get("holdout", {}) or {}).get("accuracy")
        except Exception:
            pass
        try:
            # Verification
            summary["verification_macro_auc"] = ver.get("macro_auc")
            summary["verification_macro_eer"] = ver.get("macro_eer")
        except Exception:
            pass
        try:
            # Per-position device ID
            summary["offbody_deviceid_per_position_macro_acc"] = per_pos.get("macro_acc")
        except Exception:
            pass
        try:
            # Device ID (on/off)
            summary["device_id_onbody_accuracy"] = (dev_both.get("onBody", {}) or {}).get("holdout", {}).get("accuracy")
            summary["device_id_offbody_accuracy"] = (dev_both.get("offBody", {}) or {}).get("holdout", {}).get("accuracy")
        except Exception:
            pass
        try:
            # Movement (global)
            summary["onbody_movement_global_accuracy"] = (mov.get("global", {}) or {}).get("holdout", {}).get("accuracy")
        except Exception:
            pass
        try:
            # Mixed scenario
            summary["mixed_baseline_accuracy"] = (mix.get("baseline", {}) or {}).get("holdout", {}).get("accuracy")
            summary["mixed_robust_accuracy"] = (mix.get("robust", {}) or {}).get("holdout", {}).get("accuracy")
            summary["mixed_leaky_accuracy"] = (mix.get("leaky", {}) or {}).get("holdout", {}).get("accuracy")
        except Exception:
            pass
        try:
            # Hierarchical
            hr = (hier or {}).get("hierarchical_results", {})
            summary["hierarchical_overall_accuracy"] = hr.get("overall_accuracy")
            summary["hierarchical_scenario_accuracy"] = hr.get("scenario_accuracy")
            summary["hierarchical_onbody_position_accuracy"] = hr.get("onbody_accuracy")
            summary["hierarchical_offbody_device_accuracy"] = hr.get("offbody_device_accuracy")
            summary["hierarchical_offbody_position_accuracy"] = hr.get("offbody_position_accuracy")
        except Exception:
            pass

        combined = {"summary": summary, **results}
        # Persist combined JSON
        try:
            with open(all_dir / "all_analyses_results.json", "w") as f:
                json.dump(combined, f, indent=2)
        except Exception:
            pass
        # Persist single-row CSV summary
        try:
            pd.DataFrame([summary]).to_csv(all_dir / "all_analyses_summary.csv", index=False)
        except Exception:
            pass
        return combined
    
    def _save_results(self, results: Dict[str, Any], analysis_name: str) -> Path:
        """Save analysis results to JSON file."""
        output_dir = self._get_output_dir(analysis_name)
        results_path = output_dir / f"{analysis_name}_results.json"
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        return results_path
    
    def _get_dataset_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics about a dataset."""
        if df.empty:
            return {
                "empty": True,
                "samples": 0,
                "devices": 0,
                "positions": 0,
                "scenarios": {}
            }
        
        stats = {
            "samples": len(df),
            "devices": df["dvc"].nunique() if "dvc" in df.columns else 0,
            "positions": df["pos_label"].nunique() if "pos_label" in df.columns else 0,
            "scenarios": df["scenario"].value_counts().to_dict() if "scenario" in df.columns else {}
        }
        
        if "scenario" in df.columns:
            for scenario in df["scenario"].unique():
                subset = df[df["scenario"] == scenario]
                stats[f"{scenario}_samples"] = len(subset)
                stats[f"{scenario}_devices"] = subset["dvc"].nunique() if "dvc" in subset.columns else 0
                stats[f"{scenario}_positions"] = subset["pos_label"].nunique() if "pos_label" in subset.columns else 0
        
        return stats
    
    def _generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 output_dir: Path, name: str) -> Optional[str]:
        """Generate and save confusion matrix visualization with stable labels."""
        try:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            # Normalize inputs to string arrays for consistent labeling
            y_true = np.array([str(v) for v in np.asarray(y_true)], dtype=object)
            y_pred = np.array([str(v) for v in np.asarray(y_pred)], dtype=object)
            
            # Determine a deterministic label order
            classes = np.unique(np.concatenate([y_true, y_pred]))
            classes = np.sort(classes)
            
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=(cm.shape[0] <= 10), fmt="d", cmap="Blues",
                        xticklabels=classes, yticklabels=classes)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix: {name}")
            plt.tight_layout()
            
            output_path = output_dir / f"{name}_confusion_matrix.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            return str(output_path)
        except Exception as e:
            logger.warning(f"Failed to generate confusion matrix: {e}")
            return None

    def _generate_overlay_embeddings(self, X: np.ndarray, overlays: Dict[str, np.ndarray], output_dir: Path, name_prefix: str) -> Dict[str, str]:
        """Generate PCA/UMAP 2D embeddings colored by provided overlay labels."""
        from sklearn.decomposition import PCA
        import seaborn as sns
        paths: Dict[str, str] = {}
        if X.size == 0 or X.ndim != 2:
            return paths
        # Subsample for speed
        if len(X) > 3000:
            idx = np.random.choice(len(X), 3000, replace=False)
        else:
            idx = np.arange(len(X))
        X_sub = X[idx]
        for key, vals in overlays.items():
            if vals is None:
                continue
            y_sub = np.asarray(vals)[idx]
            # PCA
            try:
                pca = PCA(n_components=2, random_state=42)
                Z = pca.fit_transform(X_sub)
                plt.figure(figsize=(8, 6))
                classes = np.unique(y_sub.astype(object))
                palette = sns.color_palette("tab10", n_colors=min(10, len(classes))) if len(classes) <= 10 else sns.color_palette("husl", len(classes))
                for i, c in enumerate(classes):
                    m = y_sub == c
                    plt.scatter(Z[m, 0], Z[m, 1], s=20, alpha=0.7, label=str(c), color=palette[i % len(palette)])
                plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
                plt.title(f"PCA by {key}: {name_prefix}")
                plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
                plt.tight_layout()
                path = output_dir / f"{name_prefix}_pca_by_{key}.png"
                plt.savefig(path, dpi=150)
                plt.close()
                paths[f"pca_by_{key}"] = str(path)
            except Exception:
                pass
            # UMAP
            try:
                import umap  # type: ignore
                reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
                Z = reducer.fit_transform(X_sub)
                plt.figure(figsize=(8, 6))
                classes = np.unique(y_sub.astype(object))
                palette = sns.color_palette("tab10", n_colors=min(10, len(classes))) if len(classes) <= 10 else sns.color_palette("husl", len(classes))
                for i, c in enumerate(classes):
                    m = y_sub == c
                    plt.scatter(Z[m, 0], Z[m, 1], s=20, alpha=0.7, label=str(c), color=palette[i % len(palette)])
                plt.title(f"UMAP by {key}: {name_prefix}")
                plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
                plt.tight_layout()
                path = output_dir / f"{name_prefix}_umap_by_{key}.png"
                plt.savefig(path, dpi=150)
                plt.close()
                paths[f"umap_by_{key}"] = str(path)
            except ModuleNotFoundError:
                logger.info("UMAP not installed. Skipping UMAP overlays. Install 'umap-learn' to enable.")
            except Exception:
                pass
        return paths

    def _select_group_column(self, df: pd.DataFrame, fallback: str = "dvc") -> str:
        """Choose a grouping column (prefer 'session' if present and informative)."""
        if "session" in df.columns and df["session"].nunique() > 1:
            return "session"
        return fallback

    def _apply_feature_ablation(self, X: np.ndarray, feature_names: List[str], ablate: Optional[Dict[str, bool]]) -> Tuple[np.ndarray, List[str]]:
        """Drop selected feature groups (CFO/RSSI/channel/meta) from X based on flags."""
        if X.size == 0 or not feature_names or not ablate:
            return X, feature_names
        drop_idx: List[int] = []
        flags = {k: bool(v) for k, v in ablate.items()}
        for i, n in enumerate(feature_names):
            # Meta groups
            if flags.get("drop_meta", False) and n.startswith("meta_"):
                drop_idx.append(i); continue
            if flags.get("drop_cfo", False) and n == "meta_cfo_hz":
                drop_idx.append(i); continue
            if flags.get("drop_rssi", False) and n == "meta_rssi_db":
                drop_idx.append(i); continue
            if flags.get("drop_channel", False) and n in ("meta_channel_sin", "meta_channel_cos"):
                drop_idx.append(i); continue
        if not drop_idx:
            return X, feature_names
        keep = np.array([i for i in range(len(feature_names)) if i not in set(drop_idx)])
        X_new = X[:, keep]
        names_new = [feature_names[i] for i in keep]
        return X_new, names_new
    
    def _extract_features(self, df: pd.DataFrame, label_col: str, 
                        remove_fingerprint: bool = False,
                        group_by: Optional[str] = None,
                        ablate: Optional[Dict[str, bool]] = None,
                        balance_by: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract features from dataframe for classification.
        
        Args:
            df: Preprocessed dataframe
            label_col: Target label column
            remove_fingerprint: Whether to remove device fingerprint (pre-feature on IQ)
            group_by: Optional grouping column (e.g., 'session')
            ablate: Optional feature ablation flags
            balance_by: Optional list of columns to balance within each class during sampling
            
        Returns:
            Dictionary containing X, y, groups, feature_names, aux_labels, class_counts
        """
        if df.empty or label_col not in df.columns:
            logger.warning(f"Empty dataframe or missing label column: {label_col}")
            return {
                "X": np.zeros((0, 0), dtype=np.float32),
                "y": np.array([]),
                "groups": np.array([]),
                "feature_names": [],
                "aux_labels": {},
                "class_counts": {}
            }
        
        # Determine effective balancing keys
        eff_balance = balance_by if balance_by is not None else self.default_balance_by
        
        # Extract features
        features = self.feature_engine.extract_features_for_training(
            df=df,
            label_col=label_col,
            max_per_class=self.max_samples_per_class,
            include_meta=True,
            advanced_features=self.use_advanced_features,
            remove_device_fingerprint=remove_fingerprint,
            group_by=group_by,
            balance_by=eff_balance
        )
        
        # Optional ablation
        X, y, groups, feature_names = features["X"], features["y"], features["groups"], features["feature_names"]
        if ablate:
            X, feature_names = self._apply_feature_ablation(X, feature_names, ablate)
        
        return {
            "X": X,
            "y": y,
            "groups": groups,
            "feature_names": feature_names,
            "aux_labels": features.get("aux_labels", {}),
            "class_counts": features.get("class_counts", {}),
            "kept_frame_indices": features.get("kept_frame_indices")
        }

    def _build_manifest_badges(self, *, task: str, remove_fp: bool, group_by: Optional[str], balance_by: Optional[List[str]], df: pd.DataFrame) -> Dict[str, Any]:
        badges = {
            "task": task,
            "fingerprint_removed_pre_feature": bool(remove_fp),
            "group_by": group_by or None,
            "balance_by": balance_by if balance_by is not None else (self.default_balance_by or []),
            "n_samples": int(len(df)),
            "has_session": bool("session" in df.columns and df["session"].nunique() > 0),
        }
        return badges

    # Helper: load sklearn pipeline from a persisted path
    def _load_model(self, path_str: Optional[str]) -> Optional[Any]:
        if not path_str or joblib is None:
            return None
        try:
            p = Path(path_str)
            if p.exists():
                return joblib.load(p)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to load model from {path_str}: {e}")
        return None
    
    def analyze_offbody_position(self, 
                               source_path: Optional[Union[str, Path]] = None, 
                               device_filter: Optional[List[int]] = None,
                               save_results: bool = True,
                               per_device: bool = True,
                               min_samples_per_cell: int = 20,
                               ablate: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        Analyze offBody position classification.
        
        Args:
            source_path: Path to data source (if None, uses data_root)
            device_filter: Optional list of device IDs to include
            save_results: Whether to save results to disk
            per_device: Additionally run per-device 7-way position classifiers
            min_samples_per_cell: Minimum samples required per (device, position)
            ablate: Optional feature ablation flags (e.g., {'drop_cfo': True})
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting offBody position analysis")
        analysis_name = "offbody_position"
        output_dir = self._get_output_dir(analysis_name)
        
        # Load offBody data
        offbody_df = self.data_manager.get_scenario_dataframe(
            scenario="offBody",
            source_path=source_path,
            preprocess_iq=True,
            max_rows=50000  # Adjust based on available memory
        )
        
        # Filter by devices if specified
        if device_filter and not offbody_df.empty:
            offbody_df = offbody_df[offbody_df["dvc"].isin(device_filter)].copy()
        
        # Log dataset info
        dataset_stats = self._get_dataset_stats(offbody_df)
        logger.info(f"Loaded {len(offbody_df)} offBody samples")
        logger.info(f"Unique positions: {dataset_stats['positions']}")
        logger.info(f"Unique devices: {dataset_stats['devices']}")
        
        if offbody_df.empty or offbody_df["pos_label"].nunique() < 2:
            return {"error": "Insufficient data", "dataset_stats": dataset_stats}
        
        # Determine grouping (prefer session if available)
        group_col = self._select_group_column(offbody_df, fallback="dvc")
        
        # Global model (remove fingerprint to focus on position)
        eff_balance = ["dvc", "session"] if "session" in offbody_df.columns else ["dvc"]
        features = self._extract_features(
            df=offbody_df, 
            label_col="pos_label", 
            remove_fingerprint=True,
            group_by=group_col,
            ablate=ablate,
            balance_by=eff_balance
        )
        
        X, y, groups = features["X"], features["y"], features["groups"]
        feature_names = features["feature_names"]
        
        results: Dict[str, Any] = {}
        if X.size == 0 or len(np.unique(y)) < 2:
            logger.error(f"Insufficient data for classification: X shape={X.shape}, classes={len(np.unique(y)) if y.size > 0 else 0}")
            results["error"] = "Insufficient data"
        else:
            manifest_extra = {
                "badges": self._build_manifest_badges(task=analysis_name, remove_fp=True, group_by=group_col, balance_by=eff_balance, df=offbody_df),
                "class_counts": features.get("class_counts", {}),
            }
            global_results = train_classifier(
                X=X,
                y=y,
                groups=groups,
                task_name=f"{analysis_name}_global",
                output_dir=output_dir,
                feature_names=feature_names,
                config={
                    "accuracy_threshold": self.accuracy_threshold,
                    "epochs": 50 if self.use_deep_learning else 0
                },
                use_grouped_split=True,
                manifest_extra=manifest_extra,
                kept_frame_indices=features.get("kept_frame_indices"),
            )
            # Chance baseline ~ 1/K
            n_pos = len(np.unique(y))
            global_results["chance_baseline"] = 1.0 / max(1, n_pos)
            results["global"] = global_results
            # Qualitative overlays: color by position, device, session after fingerprint removal
            overlays: Dict[str, np.ndarray] = {}
            aux = features.get("aux_labels", {})
            for key_src, key_dst in [("pos_label", "position"), ("dvc", "device"), ("session", "session")]:
                if aux.get(key_src, None) is not None and len(aux[key_src]) == len(y):
                    overlays[key_dst] = np.array([str(v) for v in aux[key_src]], dtype=object)
            if overlays:
                self._generate_overlay_embeddings(X, overlays, output_dir, name_prefix=f"{analysis_name}_global")
        
        # Per-device position analysis
        if per_device and not offbody_df.empty:
            per_dev: Dict[str, Any] = {}
            accs: List[float] = []
            per_rows: List[Dict[str, Any]] = []
            for dev in sorted(offbody_df["dvc"].dropna().unique()):
                ddf = offbody_df[offbody_df["dvc"] == dev].copy()
                # Data quality checks
                if ddf["pos_label"].nunique() < 2:
                    logger.info(f"Skipping device {dev}: <2 positions")
                    continue
                counts = ddf["pos_label"].value_counts()
                if int(counts.min()) < min_samples_per_cell:
                    logger.info(f"Skipping device {dev}: min samples per position {int(counts.min())} < {min_samples_per_cell}")
                    continue
                logger.info(f"Per-device position classification for device {dev} with positions={len(counts)}")
                grp_col = self._select_group_column(ddf, fallback=None) if "session" in ddf.columns else None
                feats_dev = self._extract_features(
                    df=ddf,
                    label_col="pos_label",
                    remove_fingerprint=False,  # single-device, so no fingerprint removal
                    group_by=grp_col,
                    ablate=ablate,
                    balance_by=["session"] if "session" in ddf.columns else None
                )
                Xd, yd, gd = feats_dev["X"], feats_dev["y"], feats_dev["groups"]
                if Xd.size == 0 or len(np.unique(yd)) < 2:
                    continue
                use_grouped = bool(grp_col)
                dev_out = output_dir / f"per_device/device_{dev}"
                manifest_extra = {
                    "badges": self._build_manifest_badges(task=f"{analysis_name}_dev{dev}", remove_fp=False, group_by=grp_col, balance_by=( ["session"] if "session" in ddf.columns else None ), df=ddf),
                    "class_counts": feats_dev.get("class_counts", {}),
                }
                dev_out_res = train_classifier(
                    X=Xd,
                    y=yd,
                    groups=gd,
                    task_name=f"{analysis_name}_dev{dev}",
                    output_dir=dev_out,
                    feature_names=feats_dev["feature_names"],
                    config={
                        "accuracy_threshold": self.accuracy_threshold,
                        "epochs": 15 if self.use_deep_learning else 0
                    },
                    use_grouped_split=use_grouped,
                    manifest_extra=manifest_extra,
                    kept_frame_indices=feats_dev.get("kept_frame_indices"),
                )
                # Chance baseline for this device
                n_pos_d = len(np.unique(yd))
                dev_out_res["chance_baseline"] = 1.0 / max(1, n_pos_d)
                per_dev[f"device_{dev}"] = dev_out_res
                acc = dev_out_res.get("holdout", {}).get("accuracy", None)
                if isinstance(acc, (int, float)):
                    accs.append(float(acc))
                # add row for macro table
                per_rows.append({
                    "device": dev,
                    "n_positions": int(n_pos_d),
                    "accuracy": acc,
                    "f1_weighted": (dev_out_res.get("holdout", {}) or {}).get("f1_weighted"),
                    "logo_mean": ((dev_out_res.get("logo", {}) or {}).get("accuracy_mean")),
                    "logo_folds": ((dev_out_res.get("logo", {}) or {}).get("folds")),
                    "result_run_id": dev_out_res.get("result_run_id")
                })
            if per_dev:
                results["per_device"] = per_dev
                results["per_device_macro_acc"] = float(np.mean(accs)) if accs else None
                # write per-device table
                try:
                    pd.DataFrame(per_rows).to_csv(output_dir / "per_device_summary.csv", index=False)
                except Exception:
                    pass
        
        # Add dataset stats
        results["dataset_stats"] = dataset_stats
        
        # Cache and save results
        self.results_cache[analysis_name] = results
        
        if save_results:
            self._save_results(results, analysis_name)
        
        return results
    
    def analyze_device_identification(self, 
                                    scenario: str = "both",
                                    source_path: Optional[Union[str, Path]] = None,
                                    save_results: bool = True) -> Dict[str, Any]:
        """
        Analyze device identification (fingerprinting).
        
        Args:
            scenario: 'onBody', 'offBody', or 'both'
            source_path: Path to data source (if None, uses data_root)
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting device identification analysis for scenario: {scenario}")
        
        if scenario == "both":
            # Run both scenarios and combine results
            onbody_results = self.analyze_device_identification("onBody", source_path, save_results)
            offbody_results = self.analyze_device_identification("offBody", source_path, save_results)
            
            combined_results = {
                "onBody": onbody_results,
                "offBody": offbody_results
            }
            
            # Save combined results
            if save_results:
                self._save_results(combined_results, "device_identification_combined")
            
            return combined_results
        
        analysis_name = f"{scenario.lower()}_device_identification"
        output_dir = self._get_output_dir(analysis_name)
        
        # Load data for specified scenario
        df = self.data_manager.get_scenario_dataframe(
            scenario=scenario,
            source_path=source_path,
            preprocess_iq=True,
            max_rows=50000
        )
        
        # Log dataset info
        dataset_stats = self._get_dataset_stats(df)
        logger.info(f"Loaded {len(df)} {scenario} samples")
        logger.info(f"Unique devices: {dataset_stats['devices']}")
        
        # Extract features
        eff_balance = [k for k in ["pos_label", "session"] if k in df.columns]
        features = self._extract_features(
            df=df, 
            label_col="dvc", 
            remove_fingerprint=False,  # Don't remove fingerprint for device identification
            # Balance sampling across position and session if available to reduce bias
            balance_by=eff_balance
        )
        
        X, y, groups = features["X"], features["y"], features["groups"]
        feature_names = features["feature_names"]
        
        if X.size == 0 or len(np.unique(y)) < 2:
            logger.error(f"Insufficient data for classification: X shape={X.shape}, classes={len(np.unique(y)) if y.size > 0 else 0}")
            return {"error": "Insufficient data", "dataset_stats": dataset_stats}
        
        # Train classifier
        manifest_extra = {
            "badges": self._build_manifest_badges(task=analysis_name, remove_fp=False, group_by=None, balance_by=eff_balance, df=df),
            "class_counts": features.get("class_counts", {}),
        }
        results = train_classifier(
            X=X,
            y=y,
            groups=groups,
            task_name=analysis_name,
            output_dir=output_dir,
            feature_names=feature_names,
            config={
                "accuracy_threshold": self.accuracy_threshold,
                "epochs": 50 if self.use_deep_learning else 0
            },
            use_grouped_split=False,  # Don't use grouped split for device identification
            manifest_extra=manifest_extra,
            kept_frame_indices=features.get("kept_frame_indices"),
        )
        
        # Add dataset stats
        results["dataset_stats"] = dataset_stats
        
        # Cache and save results
        self.results_cache[analysis_name] = results
        
        if save_results:
            self._save_results(results, analysis_name)
        
        return results
    
    def analyze_onbody_movement(self, 
                              per_device: bool = True,
                              source_path: Optional[Union[str, Path]] = None,
                              save_results: bool = True) -> Dict[str, Any]:
        """
        Analyze onBody movement classification.
        
        Args:
            per_device: Whether to analyze movement per device
            source_path: Path to data source (if None, uses data_root)
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting onBody movement analysis")
        analysis_name = "onbody_movement"
        output_dir = self._get_output_dir(analysis_name)
        
        # Load onBody data
        onbody_df = self.data_manager.get_scenario_dataframe(
            scenario="onBody",
            source_path=source_path,
            preprocess_iq=True,
            max_rows=50000
        )
        
        # Check if movement column exists
        if "movement" not in onbody_df.columns or onbody_df["movement"].nunique() < 2:
            logger.error("No movement data found or insufficient movement classes")
            return {"error": "No movement data", "dataset_stats": self._get_dataset_stats(onbody_df)}
        
        results = {}
        
        # Log dataset info
        dataset_stats = self._get_dataset_stats(onbody_df)
        movement_counts = onbody_df["movement"].value_counts().to_dict()
        logger.info(f"Loaded {len(onbody_df)} onBody samples")
        logger.info(f"Movement classes: {movement_counts}")
        
        # Global movement analysis (across all devices)
        logger.info("Analyzing global movement patterns (all devices)")
        
        # Prefer grouping by session when available to avoid leakage
        group_col = self._select_group_column(onbody_df, fallback="dvc")
        
        global_features = self._extract_features(
            df=onbody_df, 
            label_col="movement", 
            remove_fingerprint=True,  # Remove fingerprint to focus on movement patterns
            group_by=group_col,
            balance_by=["dvc", "session"] if "session" in onbody_df.columns else ["dvc"]
        )
        
        X, y, groups = global_features["X"], global_features["y"], global_features["groups"]
        feature_names = global_features["feature_names"]
        
        if X.size > 0 and len(np.unique(y)) >= 2:
            manifest_extra = {
                "badges": self._build_manifest_badges(task=f"{analysis_name}_global", remove_fp=True, group_by=group_col, balance_by=( ["dvc", "session"] if "session" in onbody_df.columns else ["dvc"] ), df=onbody_df),
                "class_counts": global_features.get("class_counts", {}),
            }
            global_results = train_classifier(
                X=X,
                y=y,
                groups=groups,
                task_name=f"{analysis_name}_global",
                output_dir=output_dir,
                feature_names=feature_names,
                config={
                    "accuracy_threshold": self.accuracy_threshold,
                    "epochs": 50 if self.use_deep_learning else 0
                },
                use_grouped_split=True,  # Use grouped split for device-independent evaluation
                manifest_extra=manifest_extra,
                kept_frame_indices=global_features.get("kept_frame_indices"),
            )
            
            results["global"] = global_results
            # Overlays for movement global: movement (y), device, session
            overlays_mv = {"movement": np.array([str(v) for v in y], dtype=object)}
            aux_g = global_features.get("aux_labels", {})
            for ksrc, kdst in [("dvc", "device"), ("session", "session")]:
                if aux_g.get(ksrc) is not None and len(aux_g[ksrc]) == len(y):
                    overlays_mv[kdst] = np.array([str(v) for v in aux_g[ksrc]], dtype=object)
            self._generate_overlay_embeddings(X, overlays_mv, output_dir, name_prefix=f"{analysis_name}_global")
        
        # Per-device movement analysis
        if per_device:
            logger.info("Analyzing per-device movement patterns")
            per_device_results = {}
            
            for device_id in sorted(onbody_df["dvc"].dropna().unique()):
                device_df = onbody_df[onbody_df["dvc"] == device_id].copy()
                
                if device_df["movement"].nunique() < 2 or len(device_df) < 100:
                    logger.info(f"Skipping device {device_id}: insufficient data or movement classes")
                    continue
                
                logger.info(f"Analyzing movement for device {device_id}")
                
                device_features = self._extract_features(
                    df=device_df, 
                    label_col="movement", 
                    remove_fingerprint=False,  # No need to remove fingerprint for single device
                    balance_by=["session"] if "session" in device_df.columns else None
                )
                
                X_dev, y_dev, g_dev = device_features["X"], device_features["y"], device_features["groups"]
                
                if X_dev.size > 0 and len(np.unique(y_dev)) >= 2:
                    manifest_extra = {
                        "badges": self._build_manifest_badges(task=f"{analysis_name}_device{device_id}", remove_fp=False, group_by=None, balance_by=( ["session"] if "session" in device_df.columns else None ), df=device_df),
                        "class_counts": device_features.get("class_counts", {}),
                    }
                    device_result = train_classifier(
                        X=X_dev,
                        y=y_dev,
                        groups=g_dev,
                        task_name=f"{analysis_name}_device{device_id}",
                        output_dir=output_dir / f"device_{device_id}",
                        feature_names=device_features["feature_names"],
                        config={
                            "accuracy_threshold": self.accuracy_threshold,
                            "epochs": 15 if self.use_deep_learning else 0
                        },
                        use_grouped_split=False,  # No need for grouped split within a device
                        manifest_extra=manifest_extra,
                        kept_frame_indices=device_features.get("kept_frame_indices"),
                    )
                    # Overlays: session for per-device movement
                    aux_dev = device_features.get("aux_labels", {})
                    overlays_dev = {}
                    if aux_dev.get("session") is not None and len(aux_dev["session"]) == len(y_dev):
                        overlays_dev["session"] = np.array([str(v) for v in aux_dev["session"]], dtype=object)
                    if overlays_dev:
                        self._generate_overlay_embeddings(X_dev, overlays_dev, output_dir / f"device_{device_id}", name_prefix=f"{analysis_name}_device{device_id}")
                    
                    per_device_results[f"device_{device_id}"] = device_result
            
            results["per_device"] = per_device_results
        
        # Add dataset stats
        results["dataset_stats"] = dataset_stats
        
        # Cache and save results
        self.results_cache[analysis_name] = results
        
        if save_results:
            self._save_results(results, analysis_name)
        
        return results
    
    def analyze_mixed_scenario(self,
                             source_path: Optional[Union[str, Path]] = None,
                             save_results: bool = True) -> Dict[str, Any]:
        """
        Analyze mixed scenario classification (onBody vs offBody).
        
        Args:
            source_path: Path to data source (if None, uses data_root)
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting mixed scenario analysis (onBody vs offBody)")
        analysis_name = "mixed_scenario"
        output_dir = self._get_output_dir(analysis_name)
        
        # Load mixed scenario data
        mixed_df = self.data_manager.get_mixed_scenario_dataframe(
            source_path=source_path,
            max_per_scenario=self.max_samples_per_class
        )
        
        # Log dataset info
        dataset_stats = self._get_dataset_stats(mixed_df)
        logger.info(f"Loaded {len(mixed_df)} mixed scenario samples")
        logger.info(f"onBody samples: {dataset_stats.get('onBody_samples', 0)}")
        logger.info(f"offBody samples: {dataset_stats.get('offBody_samples', 0)}")
        
        # Prefer grouping by session when available
        group_col = self._select_group_column(mixed_df, fallback="dvc")
        eff_balance = [k for k in ["dvc", "pos_label", "session"] if k in mixed_df.columns]
        
        # Baseline features (with device fingerprint)
        feats_base = self._extract_features(
            df=mixed_df, 
            label_col="scenario", 
            remove_fingerprint=False,
            group_by=group_col,
            balance_by=eff_balance
        )
        # Robust features (pre-feature fingerprint removal)
        feats_rob = self._extract_features(
            df=mixed_df, 
            label_col="scenario", 
            remove_fingerprint=True,
            group_by=group_col,
            balance_by=eff_balance
        )
        
        Xb, yb, gb = feats_base["X"], feats_base["y"], feats_base["groups"]
        Xr, yr, gr = feats_rob["X"], feats_rob["y"], feats_rob["groups"]
        feature_names = feats_base["feature_names"]
        
        if Xb.size == 0 or len(np.unique(yb)) < 2:
            logger.error(f"Insufficient data for classification: X shape={Xb.shape}, classes={len(np.unique(yb)) if yb.size > 0 else 0}")
            return {"error": "Insufficient data", "dataset_stats": dataset_stats}
        
        # Train with device fingerprint (baseline)
        logger.info("Training scenario classifier with device fingerprints")
        manifest_base = {
            "badges": self._build_manifest_badges(task=f"{analysis_name}_baseline", remove_fp=False, group_by=group_col, balance_by=eff_balance, df=mixed_df),
            "class_counts": feats_base.get("class_counts", {}),
        }
        baseline_results = train_classifier(
            X=Xb,
            y=yb,
            groups=gb,
            task_name=f"{analysis_name}_baseline",
            output_dir=output_dir / "baseline",
            feature_names=feature_names,
            config={
                "accuracy_threshold": self.accuracy_threshold,
                "epochs": 50 if self.use_deep_learning else 0
            },
            use_grouped_split=True,  # Use grouped split for device-independent evaluation
            manifest_extra=manifest_base,
            kept_frame_indices=feats_base.get("kept_frame_indices"),
        )
        
        # Train without device fingerprint (robust) using pre-feature mitigation
        logger.info("Training scenario classifier without device fingerprints (pre-feature mitigation)")
        manifest_rob = {
            "badges": self._build_manifest_badges(task=f"{analysis_name}_robust", remove_fp=True, group_by=group_col, balance_by=eff_balance, df=mixed_df),
            "class_counts": feats_rob.get("class_counts", {}),
        }
        robust_results = train_classifier(
            X=Xr,
            y=yr,
            groups=gr,
            task_name=f"{analysis_name}_robust",
            output_dir=output_dir / "robust",
            feature_names=feature_names,
            config={
                "accuracy_threshold": self.accuracy_threshold,
                "epochs": 50 if self.use_deep_learning else 0
            },
            use_grouped_split=True,
            manifest_extra=manifest_rob,
            kept_frame_indices=feats_rob.get("kept_frame_indices"),
        )
        
        # Qualitative overlays using robust features: scenario/device/position/session
        try:
            overlays: Dict[str, np.ndarray] = {}
            overlays["scenario"] = np.array([str(v).replace("scenario_", "") for v in yr], dtype=object)
            aux = feats_rob.get("aux_labels", {})
            for src, dst in [("dvc", "device"), ("pos_label", "position"), ("session", "session")]:
                if aux.get(src) is not None and len(aux[src]) == len(yr):
                    overlays[dst] = np.array([str(v) for v in aux[src]], dtype=object)
            if overlays:
                self._generate_overlay_embeddings(Xr, overlays, output_dir / "robust", name_prefix=f"{analysis_name}_robust")
        except Exception as e:
            logger.warning(f"Overlay generation failed for mixed scenario: {e}")
        
        # Evaluate on leaky split (non-grouped, same devices in train and test)
        logger.info("Training scenario classifier with leaky device split")
        manifest_leaky = {
            "badges": self._build_manifest_badges(task=f"{analysis_name}_leaky", remove_fp=False, group_by=None, balance_by=eff_balance, df=mixed_df),
            "class_counts": feats_base.get("class_counts", {}),
        }
        leaky_results = train_classifier(
            X=Xb,
            y=yb,
            groups=gb,
            task_name=f"{analysis_name}_leaky",
            output_dir=output_dir / "leaky",
            feature_names=feature_names,
            config={
                "accuracy_threshold": self.accuracy_threshold,
                "epochs": 50 if self.use_deep_learning else 0
            },
            use_grouped_split=False,  # Don't use grouped split (leaky)
            manifest_extra=manifest_leaky,
            kept_frame_indices=feats_base.get("kept_frame_indices"),
        )
        
        # Combine results
        results = {
            "baseline": baseline_results,
            "robust": robust_results,
            "leaky": leaky_results,
            "dataset_stats": dataset_stats
        }
        
        # Cache and save results
        self.results_cache[analysis_name] = results
        
        if save_results:
            self._save_results(results, analysis_name)
        
        return results
    
    def analyze_hierarchical_classification(self,
                                          source_path: Optional[Union[str, Path]] = None,
                                          save_results: bool = True) -> Dict[str, Any]:
        """
        Analyze hierarchical classification:
        1. First classify scenario (onBody vs offBody)
        2a. If onBody: use device prediction as position
        2b. If offBody: classify device, then position
        
        Args:
            source_path: Path to data source (if None, uses data_root)
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting hierarchical classification analysis")
        analysis_name = "hierarchical_classification"
        output_dir = self._get_output_dir(analysis_name)
        # per-run data pack
        run_id = f"{analysis_name}_{int(time.time())}"
        pack_dir = output_dir / f"{analysis_name}__{run_id}"
        pack_dir.mkdir(parents=True, exist_ok=True)
        
        # First, make sure we have scenario, device, and position classifiers
        mixed_results = self.analyze_mixed_scenario(source_path, save_results=False)
        device_results = self.analyze_device_identification("both", source_path, save_results=False)
        offbody_position_results = self.analyze_offbody_position(source_path, save_results=False)
        
        # Load mixed scenario data
        mixed_df = self.data_manager.get_mixed_scenario_dataframe(
            source_path=source_path,
            max_per_scenario=self.max_samples_per_class
        )
        
        # For hierarchical evaluation, we need to split the data differently
        # to have independent test sets for each level
        from sklearn.model_selection import train_test_split, GroupShuffleSplit
        
        # Split data ensuring device separation
        if "dvc" in mixed_df.columns and mixed_df["dvc"].nunique() > 3:
            # Use grouped split to separate devices
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            
            try:
                train_idx, test_idx = next(splitter.split(mixed_df, groups=mixed_df["dvc"]))
                train_df = mixed_df.iloc[train_idx].copy()
                test_df = mixed_df.iloc[test_idx].copy()
            except Exception as e:
                logger.warning(f"Grouped split failed: {e}. Falling back to random split.")
                train_df, test_df = train_test_split(mixed_df, test_size=0.3, random_state=42)
        else:
            train_df, test_df = train_test_split(mixed_df, test_size=0.3, random_state=42)
        
        # Extract features from test set for evaluation
        test_features = self._extract_features(
            df=test_df, 
            label_col="scenario"
        )
        
        X_test, y_test, g_test = test_features["X"], test_features["y"], test_features["groups"]
        
        if X_test.size == 0 or len(np.unique(y_test)) < 2:
            logger.error("Insufficient test data for hierarchical evaluation")
            return {"error": "Insufficient test data", "component_results": {
                "mixed_scenario": mixed_results,
                "device_identification": device_results,
                "offbody_position": offbody_position_results
            }}
        
        # 1. Scenario Prediction
        # Use best model from mixed scenario analysis for prediction (baseline model)
        # Prefer loading from persisted model_path for compatibility
        scenario_model = None
        try:
            scenario_model = self._load_model((mixed_results.get("baseline", {}) or {}).get("model_path"))
        except Exception:
            scenario_model = None
        
        if scenario_model is None:
            logger.error("No scenario model available for hierarchical classification")
            return {"error": "No scenario model", "component_results": {
                "mixed_scenario": mixed_results,
                "device_identification": device_results,
                "offbody_position": offbody_position_results
            }}
        
        # Predict scenarios
        scenario_predictions = scenario_model.predict(X_test)
        
        # Also compute scenario ROC/ECE and confusion (head diagnostics)
        head_artifacts: Dict[str, Any] = {}
        try:
            from sklearn.metrics import roc_curve, auc, confusion_matrix
            # ROC (binary onBody/offBody)
            if hasattr(scenario_model, "predict_proba"):
                proba = scenario_model.predict_proba(X_test)
                classes = None
                if hasattr(scenario_model, "steps"):
                    classes = scenario_model.steps[-1][1].classes_
                elif hasattr(scenario_model, "classes_"):
                    classes = scenario_model.classes_
                if classes is not None and len(classes) == 2:
                    cls_to_idx = {str(c): i for i, c in enumerate(classes)}
                    y_idx = np.array([cls_to_idx.get(str(v), -1) for v in y_test])
                    mask = y_idx >= 0
                    if np.any(mask):
                        fpr, tpr, thr = roc_curve(y_idx[mask], proba[mask, 1])
                        auc_val = float(auc(fpr, tpr))
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(6, 5))
                        plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
                        plt.plot([0, 1], [0, 1], 'k--')
                        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC: scenario head")
                        plt.legend(loc="lower right"); plt.tight_layout()
                        roc_png = pack_dir / "scenario_head_roc.png"
                        plt.savefig(roc_png, dpi=150); plt.close()
                        roc_csv = pack_dir / "scenario_head_roc.csv"
                        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(roc_csv, index=False)
                        head_artifacts["roc_png"] = str(roc_png)
                        head_artifacts["roc_csv"] = str(roc_csv)
                        
                        # Reliability (ECE bins)
                        # compute bins over max-confidence
                        y_true_arr = np.array([str(v) for v in y_test], dtype=object)
                        label_to_idx = {str(c): i for i, c in enumerate(classes)}
                        y_idx_all = np.array([label_to_idx.get(str(v), -1) for v in y_true_arr])
                        m2 = y_idx_all >= 0
                        if np.any(m2):
                            conf = np.max(proba[m2], axis=1)
                            pred_idx = np.argmax(proba[m2], axis=1)
                            corr = (pred_idx == y_idx_all[m2]).astype(float)
                            bins = np.linspace(0.0, 1.0, 16)
                            accs, confs, counts = [], [], []
                            for i in range(len(bins) - 1):
                                m = (conf >= bins[i]) & (conf < bins[i + 1] if i < len(bins) - 2 else conf <= bins[i + 1])
                                if not np.any(m):
                                    accs.append(np.nan); confs.append(np.nan); counts.append(0)
                                else:
                                    accs.append(float(np.mean(corr[m]))); confs.append(float(np.mean(conf[m]))); counts.append(int(np.sum(m)))
                            # plot
                            plt.figure(figsize=(6, 5))
                            plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
                            plt.plot([c for c in confs if np.isfinite(c)], [a for a in accs if np.isfinite(a)], marker='o', label='Empirical')
                            plt.xlabel('Confidence'); plt.ylabel('Accuracy'); plt.title('Reliability: scenario head')
                            plt.legend(loc='best'); plt.tight_layout()
                            rel_png = pack_dir / "scenario_head_reliability.png"
                            plt.savefig(rel_png, dpi=150); plt.close()
                            rel_csv = pack_dir / "scenario_head_reliability_bins.csv"
                            pd.DataFrame({
                                "bin_lower": bins[:-1],
                                "bin_upper": bins[1:],
                                "avg_conf": confs,
                                "acc": accs,
                                "count": counts
                            }).to_csv(rel_csv, index=False)
                            head_artifacts["reliability_png"] = str(rel_png)
                            head_artifacts["reliability_bins_csv"] = str(rel_csv)
            # Scenario confusion matrix (PNG+CSV)
            from sklearn.metrics import confusion_matrix
            y_true_s = np.array([str(v) for v in y_test], dtype=object)
            y_pred_s = np.array([str(v) for v in scenario_predictions], dtype=object)
            classes_s = np.sort(np.unique(np.concatenate([y_true_s, y_pred_s])))
            cm = confusion_matrix(y_true_s, y_pred_s, labels=classes_s)
            import seaborn as sns
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True if cm.shape[0] <= 10 else False, fmt="d", cmap="Blues",
                        xticklabels=classes_s, yticklabels=classes_s)
            plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Scenario Confusion")
            plt.tight_layout()
            cm_png = pack_dir / "scenario_head_confusion.png"
            plt.savefig(cm_png, dpi=150); plt.close()
            cm_csv = pack_dir / "scenario_head_confusion.csv"
            pd.DataFrame(cm, index=classes_s, columns=classes_s).to_csv(cm_csv)
            head_artifacts["confusion_png"] = str(cm_png)
            head_artifacts["confusion_csv"] = str(cm_csv)
        except Exception as e:
            logger.warning(f"Scenario head diagnostics failed: {e}")
        
        # Helper to strip known prefixes from encoded labels
        def _strip_prefix(val: Any, prefix: str) -> str:
            s = str(val)
            return s[len(prefix):] if s.startswith(prefix) else s
        
        # Prepare hierarchical results container
        hierarchical_results = {
            "scenario_accuracy": 0.0,
            "onbody_accuracy": 0.0,
            "offbody_device_accuracy": 0.0,
            "offbody_position_accuracy": 0.0,
            "overall_accuracy": 0.0,
            "predictions": [],
            "ground_truth": []
        }
        
        # 2. Split based on predicted scenario
        pred_scenarios = np.array([_strip_prefix(v, "scenario_") for v in scenario_predictions], dtype=object)
        true_scenarios = np.array([_strip_prefix(v, "scenario_") for v in y_test], dtype=object)
        onbody_mask = pred_scenarios == "onBody"
        offbody_mask = ~onbody_mask
        
        # Check if we have any predicted onBody and offBody samples
        if not onbody_mask.any() or not offbody_mask.any():
            logger.error("Scenario classifier predicted only one class, hierarchical classification not possible")
            return {"error": "Imbalanced scenario predictions", "component_results": {
                "mixed_scenario": mixed_results,
                "device_identification": device_results,
                "offbody_position": offbody_position_results
            }}
        
        # 3a. For predicted onBody samples, use device ID as position (1:1 mapping)
        onbody_device_model = None
        try:
            onbody_device_model = self._load_model((device_results.get("onBody", {}) or {}).get("model_path"))
        except Exception:
            onbody_device_model = None
        # 3b. OffBody models
        offbody_device_model = None
        try:
            offbody_device_model = self._load_model((device_results.get("offBody", {}) or {}).get("model_path"))
        except Exception:
            offbody_device_model = None
        offbody_position_model = None
        try:
            offbody_position_model = self._load_model((offbody_position_results.get("global", {}) or {}).get("model_path"))
        except Exception:
            offbody_position_model = None
        
        # Predict and evaluate
        # First, evaluate scenario classification with normalized labels
        scenario_accuracy = np.mean(pred_scenarios == true_scenarios)
        hierarchical_results["scenario_accuracy"] = float(scenario_accuracy)
        
        # Now evaluate device and position classification
        predicted_positions: List[Any] = []
        true_positions = test_df["pos_label"].values
        
        # Build device->position map from onBody training data using normalized device labels
        device_to_position: Dict[str, Any] = {}
        try:
            onbody_train = train_df[train_df["scenario"] == "onBody"]
        except Exception:
            onbody_train = train_df[train_df["scenario"].astype(str).str.endswith("onBody")]
        for _, row in onbody_train.iterrows():
            device_to_position[str(row["dvc"])] = row["pos_label"]
        
        # For onBody samples
        X_onbody = X_test[onbody_mask]
        if X_onbody.size > 0 and onbody_device_model is not None:
            onbody_device_predictions = onbody_device_model.predict(X_onbody)
            onbody_dev_norm = [_strip_prefix(dev, "dvc_") for dev in onbody_device_predictions]
            onbody_position_predictions = [device_to_position.get(str(dev), "unknown") for dev in onbody_dev_norm]
            onbody_true_positions = true_positions[onbody_mask]
            onbody_accuracy = np.mean([p == t for p, t in zip(onbody_position_predictions, onbody_true_positions)])
            hierarchical_results["onbody_accuracy"] = float(onbody_accuracy)
            predicted_positions.extend(onbody_position_predictions)
        
        # For offBody samples
        X_offbody = X_test[offbody_mask]
        if X_offbody.size > 0 and offbody_device_model is not None and offbody_position_model is not None:
            offbody_device_predictions = offbody_device_model.predict(X_offbody)
            offbody_dev_norm = [_strip_prefix(dev, "dvc_") for dev in offbody_device_predictions]
            offbody_true_devices = test_df["dvc"].values[offbody_mask]
            offbody_device_accuracy = np.mean([str(p) == str(t) for p, t in zip(offbody_dev_norm, offbody_true_devices)])
            hierarchical_results["offbody_device_accuracy"] = float(offbody_device_accuracy)
            
            offbody_position_predictions = offbody_position_model.predict(X_offbody)
            offbody_pos_norm = [_strip_prefix(p, "pos_label_") for p in offbody_position_predictions]
            offbody_true_positions = true_positions[offbody_mask]
            offbody_position_accuracy = np.mean([p == t for p, t in zip(offbody_pos_norm, offbody_true_positions)])
            hierarchical_results["offbody_position_accuracy"] = float(offbody_position_accuracy)
            predicted_positions.extend(offbody_pos_norm)
        
        # Calculate overall position accuracy
        overall_accuracy = np.mean([p == t for p, t in zip(predicted_positions, true_positions)])
        hierarchical_results["overall_accuracy"] = float(overall_accuracy)
        
        # Store predictions and ground truth for visualization
        hierarchical_results["predictions"] = predicted_positions
        hierarchical_results["ground_truth"] = true_positions.tolist()
        
        # Generate confusion matrix
        self._generate_confusion_matrix(
            true_positions,
            predicted_positions,
            output_dir,
            "hierarchical_position"
        )
        # Also save confusion CSV for positions
        try:
            from sklearn.metrics import confusion_matrix
            import pandas as pd
            y_true_cm = np.array([str(v) for v in true_positions], dtype=object)
            y_pred_cm = np.array([str(v) for v in predicted_positions], dtype=object)
            classes_cm = np.sort(np.unique(np.concatenate([y_true_cm, y_pred_cm])))
            cm_pos = confusion_matrix(y_true_cm, y_pred_cm, labels=classes_cm)
            pd.DataFrame(cm_pos, index=classes_cm, columns=classes_cm).to_csv(pack_dir / "hierarchical_position_confusion.csv")
        except Exception:
            pass
        
        # Persist summary CSV and manifest
        try:
            summary_row = {"result_run_id": run_id, **{k: hierarchical_results.get(k) for k in [
                "scenario_accuracy", "onbody_accuracy", "offbody_device_accuracy", "offbody_position_accuracy", "overall_accuracy"
            ]},
            # include branch counts for paper tables
            "n_pred_onbody": int(np.sum(onbody_mask)),
            "n_pred_offbody": int(np.sum(offbody_mask))}
            pd.DataFrame([summary_row]).to_csv(pack_dir / "hierarchical_summary.csv", index=False)
        except Exception:
            pass
        # Also persist a compact manifest for the hierarchical run
        try:
            manifest = {
                "result_run_id": run_id,
                "task": analysis_name,
                "summary_csv": str(pack_dir / "hierarchical_summary.csv"),
                "head_artifacts": head_artifacts,
                "branch_counts": {
                    "n_pred_onbody": int(np.sum(onbody_mask)),
                    "n_pred_offbody": int(np.sum(offbody_mask))
                },
                "position_confusion_png": str(pack_dir / "hierarchical_position_confusion.png"),
                "position_confusion_csv": str(pack_dir / "hierarchical_position_confusion.csv"),
                # References to component analyses for traceability
                "components": {
                    "mixed_scenario": (mixed_results.get("baseline", {}) or {}).get("result_run_id"),
                    "device_onbody": (device_results.get("onBody", {}) or {}).get("result_run_id"),
                    "device_offbody": (device_results.get("offBody", {}) or {}).get("result_run_id"),
                    "offbody_position_global": (offbody_position_results.get("global", {}) or {}).get("result_run_id"),
                }
            }
            with open(pack_dir / "data_pack.json", "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass
        
        # Add component results for reference
        results = {
            "result_run_id": run_id,
            "data_pack_dir": str(pack_dir),
            "hierarchical_results": hierarchical_results,
            "head_artifacts": head_artifacts,
            "component_results": {
                "mixed_scenario": mixed_results,
                "device_identification": device_results,
                "offbody_position": offbody_position_results
            }
        }
        
        # Cache and save results
        self.results_cache[analysis_name] = results
        
        if save_results:
            self._save_results(results, analysis_name)
        
        return results

    # ------------------------------
    # New: OffBody verification and per-position device ID analyses
    # ------------------------------
    def analyze_offbody_verification(
        self,
        source_path: Optional[Union[str, Path]] = None,
        device_filter: Optional[List[int]] = None,
        save_results: bool = True,
        min_samples_per_cell: int = 100,
        ablate: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """Verification-style per-device position analysis using pairwise distances.
        Reports ROC/AUC/EER per device (same vs different position).
        """
        analysis_name = "offbody_verification"
        out_dir = self._get_output_dir(analysis_name)
        # per-run data pack
        run_id = f"{analysis_name}_{int(time.time())}"
        pack_dir = out_dir / f"{analysis_name}__{run_id}"
        pack_dir.mkdir(parents=True, exist_ok=True)
        df = self.data_manager.get_scenario_dataframe(
            scenario="offBody",
            source_path=source_path,
            preprocess_iq=True,
            max_rows=50000
        )
        if df.empty or df.get("pos_label", pd.Series()).nunique() < 2:
            return {"error": "Insufficient offBody data"}
        if device_filter:
            df = df[df["dvc"].isin(device_filter)].copy()
        group_col = self._select_group_column(df, fallback=None)
        results: Dict[str, Any] = {"per_device": {}, "result_run_id": run_id, "data_pack_dir": str(pack_dir)}
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        
        summary_rows: List[Dict[str, Any]] = []
        per_dev_dir = pack_dir / "devices"
        per_dev_dir.mkdir(parents=True, exist_ok=True)
        
        for dev in sorted(df["dvc"].dropna().unique()):
            ddf = df[df["dvc"] == dev].copy()
            if ddf["pos_label"].nunique() < 2:
                continue
            counts = ddf["pos_label"].value_counts()
            if int(counts.min()) < min_samples_per_cell:
                logger.info(f"Skipping device {dev}: min samples per position {int(counts.min())} < {min_samples_per_cell}")
                continue
            feats = self._extract_features(
                ddf,
                label_col="pos_label",
                remove_fingerprint=False,
                group_by=group_col,
                ablate=ablate
            )
            X, y, g = feats["X"], feats["y"], feats["groups"]
            if X.size == 0 or len(np.unique(y)) < 2:
                continue
            # Build pairs: prefer cross-session pairs if session available
            n = len(y)
            if n < 4:
                continue
            # Indices by position (class)
            idx_by_pos: Dict[str, np.ndarray] = {}
            for cls in np.unique(y):
                idx = np.where(y == cls)[0]
                if idx.size >= 2:
                    idx_by_pos[str(cls)] = idx
            if len(idx_by_pos) < 2:
                continue
            # Create pair lists
            scores: List[float] = []
            labels: List[int] = []
            # Helper to check cross-session
            def cross_session(i: int, j: int) -> bool:
                if g is None or len(g) == 0:
                    return True
                return str(g[i]) != str(g[j]) if group_col else True
            # Same-position pairs
            for cls, idxs in idx_by_pos.items():
                # sample up to 200 pairs per class
                pairs = []
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        if cross_session(idxs[i], idxs[j]):
                            pairs.append((idxs[i], idxs[j]))
                if len(pairs) > 200:
                    rng = np.random.default_rng(42)
                    pairs = list(rng.choice(pairs, 200, replace=False))
                for i, j in pairs:
                    d = np.linalg.norm(X[i] - X[j])
                    scores.append(-float(d))  # higher score => more similar
                    labels.append(1)
            # Different-position pairs
            classes = list(idx_by_pos.keys())
            for a in range(len(classes)):
                for b in range(a + 1, len(classes)):
                    ia = idx_by_pos[classes[a]]
                    ib = idx_by_pos[classes[b]]
                    pairs = []
                    for i in ia:
                        for j in ib:
                            if cross_session(i, j):
                                pairs.append((i, j))
                    # sample to balance
                    max_pairs = 400
                    if len(pairs) > max_pairs:
                        rng = np.random.default_rng(43)
                        pairs = list(rng.choice(pairs, max_pairs, replace=False))
                    for i, j in pairs:
                        d = np.linalg.norm(X[i] - X[j])
                        scores.append(-float(d))
                        labels.append(0)
            if not scores or not labels:
                continue
            fpr, tpr, thr = roc_curve(labels, scores)
            auc_val = float(auc(fpr, tpr))
            # EER computation: find point where FPR ~= 1-TPR
            fnr = 1 - tpr
            # Find minimal difference
            idx_eer = int(np.argmin(np.abs(fpr - fnr)))
            eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2.0)
            tau_eer = float(thr[idx_eer]) if idx_eer < len(thr) else None
            # Thresholds at target FARs
            def _tau_at(target: float) -> Optional[float]:
                if fpr.size == 0:
                    return None
                k = int(np.argmin(np.abs(fpr - target)))
                return float(thr[k]) if k < len(thr) else None
            tau_far_1 = _tau_at(0.01)
            tau_far_5 = _tau_at(0.05)
            # Plot ROC
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC (device {dev})")
            plt.legend(loc="lower right")
            plt.tight_layout()
            # Save under per-run pack
            dev_dir = per_dev_dir / f"device_{dev}"
            dev_dir.mkdir(parents=True, exist_ok=True)
            roc_path = dev_dir / "roc.png"
            plt.savefig(roc_path, dpi=150)
            plt.close()
            # Save ROC CSV
            roc_csv = dev_dir / "roc.csv"
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(roc_csv, index=False)
            # Save thresholds JSON
            thres_json = dev_dir / "thresholds.json"
            with open(thres_json, "w") as f:
                json.dump({"tau_eer": tau_eer, "tau_far_1pct": tau_far_1, "tau_far_5pct": tau_far_5}, f, indent=2)
            
            results["per_device"][f"device_{dev}"] = {
                "auc": auc_val,
                "eer": eer,
                "tau_eer": tau_eer,
                "tau_far_1pct": tau_far_1,
                "tau_far_5pct": tau_far_5,
                "roc_curve": str(roc_path),
                "roc_csv": str(roc_csv),
                "thresholds_json": str(thres_json),
                "n_pairs": len(labels)
            }
            summary_rows.append({
                "result_run_id": run_id,
                "device": dev,
                "auc": auc_val,
                "eer": eer,
                "tau_eer": tau_eer,
                "tau_far_1pct": tau_far_1,
                "tau_far_5pct": tau_far_5,
                "n_pairs": len(labels),
                "roc_png": str(roc_path),
                "roc_csv": str(roc_csv)
            })
        # Macro
        aucs = [v["auc"] for v in results["per_device"].values()] or [0.0]
        eers = [v["eer"] for v in results["per_device"].values()] or [1.0]
        results["macro_auc"] = float(np.mean(aucs))
        results["macro_eer"] = float(np.mean(eers))
        # EER histogram
        try:
            if results["per_device"]:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 4))
                plt.hist(list(eers), bins=10, range=(0, 1), color="#4285F4", edgecolor="#1a237e")
                plt.xlabel("EER"); plt.ylabel("Count"); plt.title("EER Histogram (per device)")
                plt.tight_layout()
                eer_hist = pack_dir / "eer_histogram.png"
                plt.savefig(eer_hist, dpi=150)
                plt.close()
                results["eer_histogram"] = str(eer_hist)
        except Exception:
            pass
        # Fallback: retry with lower threshold if nothing passed
        if not results["per_device"] and min_samples_per_cell > 20:
            logger.info(f"No devices passed min_samples_per_cell={min_samples_per_cell}. Retrying with 20.")
            return self.analyze_offbody_verification(
                source_path=source_path,
                device_filter=device_filter,
                save_results=save_results,
                min_samples_per_cell=20,
                ablate=ablate,
            )
        # Write table-ready summary CSV
        try:
            if summary_rows:
                # add macro row at top
                macro_row = {
                    "result_run_id": run_id,
                    "device": "macro",
                    "auc": results.get("macro_auc"),
                    "eer": results.get("macro_eer"),
                    "n_pairs": sum(r.get("n_pairs", 0) for r in summary_rows)
                }
                df_sum = pd.DataFrame([macro_row] + summary_rows)
                sum_csv = pack_dir / "verification_summary.csv"
                df_sum.to_csv(sum_csv, index=False)
                results["summary_csv"] = str(sum_csv)
        except Exception:
            pass
        if save_results:
            self._save_results(results, analysis_name)
        return results

    def analyze_offbody_device_id_per_position(
        self,
        source_path: Optional[Union[str, Path]] = None,
        save_results: bool = True,
        min_samples_per_cell: int = 100,
        ablate: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """Per-position device identification on offBody samples, grouped by session if available."""
        analysis_name = "offbody_deviceid_per_position"
        out_dir = self._get_output_dir(analysis_name)
        df = self.data_manager.get_scenario_dataframe(
            scenario="offBody",
            source_path=source_path,
            preprocess_iq=True,
            max_rows=50000
        )
        if df.empty or "pos_label" not in df.columns or df["pos_label"].nunique() < 1:
            return {"error": "Insufficient offBody data"}
        group_col = self._select_group_column(df, fallback="dvc")
        results: Dict[str, Any] = {"per_position": {}}
        macro_accs: List[float] = []
        table_rows: List[Dict[str, Any]] = []
        for pos in sorted(df["pos_label"].dropna().unique()):
            pdf = df[df["pos_label"] == pos].copy()
            if pdf.get("dvc", pd.Series()).nunique() < 2:
                continue
            # Quality check: enough per-device samples
            counts = pdf["dvc"].value_counts()
            if int(counts.min()) < min_samples_per_cell:
                logger.info(f"Skipping position {pos}: min samples per device {int(counts.min())} < {min_samples_per_cell}")
                continue
            feats = self._extract_features(
                pdf,
                label_col="dvc",
                remove_fingerprint=False,
                group_by=group_col,
                ablate=ablate,
                balance_by=["session"] if "session" in pdf.columns else None
            )
            X, y, g = feats["X"], feats["y"], feats["groups"]
            if X.size == 0 or len(np.unique(y)) < 2:
                continue
            manifest_extra = {
                "badges": self._build_manifest_badges(task=f"{analysis_name}_{pos}", remove_fp=False, group_by=group_col, balance_by=( ["session"] if "session" in pdf.columns else None ), df=pdf),
                "class_counts": feats.get("class_counts", {}),
            }
            res = train_classifier(
                X=X,
                y=y,
                groups=g,
                task_name=f"{analysis_name}_{pos}",
                output_dir=out_dir / f"pos_{pos}",
                feature_names=feats["feature_names"],
                config={
                    "accuracy_threshold": self.accuracy_threshold,
                    "epochs": 30 if self.use_deep_learning else 0
                },
                use_grouped_split=True,
                manifest_extra=manifest_extra,
                kept_frame_indices=feats.get("kept_frame_indices"),
            )
            n_dev = len(np.unique(y))
            res["chance_baseline"] = 1.0 / max(1, n_dev)
            # Diagnostics: class counts and group coverage
            res["class_counts"] = feats.get("class_counts", {})
            aux = feats.get("aux_labels", {})
            grp_cov = {}
            # Determine group field
            grp_field = "session" if "session" in aux else None
            if grp_field is not None:
                sessions = aux[grp_field]
                for cls in np.unique(y):
                    m = y == cls
                    grp_cov[str(cls)] = int(len(np.unique(sessions[m])))
            res["group_coverage_unique_groups_per_class"] = grp_cov
            # Overlays: device (y) and session if available
            overlays = {"device": np.array([str(v) for v in y], dtype=object)}
            if grp_field is not None and len(aux[grp_field]) == len(y):
                overlays["session"] = np.array([str(v) for v in aux[grp_field]], dtype=object)
            self._generate_overlay_embeddings(X, overlays, out_dir / f"pos_{pos}", name_prefix=f"{analysis_name}_{pos}")
            
            results["per_position"][str(pos)] = res
            # Aggregate macro accuracy
            acc = res.get("holdout", {}).get("accuracy", None)
            if isinstance(acc, (int, float)):
                macro_accs.append(float(acc))
            table_rows.append({
                "position": pos,
                "n_devices": int(n_dev),
                "accuracy": acc,
                "f1_weighted": (res.get("holdout", {}) or {}).get("f1_weighted"),
                "logo_mean": ((res.get("logo", {}) or {}).get("accuracy_mean")),
                "logo_folds": ((res.get("logo", {}) or {}).get("folds")),
                "result_run_id": res.get("result_run_id")
            })
        results["macro_acc"] = float(np.mean(macro_accs)) if macro_accs else None
        # Fallback: retry with lower threshold if nothing passed
        if not results["per_position"] and min_samples_per_cell > 20:
            logger.info(f"No positions passed min_samples_per_cell={min_samples_per_cell}. Retrying with 20.")
            return self.analyze_offbody_device_id_per_position(
                source_path=source_path,
                save_results=save_results,
                min_samples_per_cell=20,
                ablate=ablate,
            )
        # Save per-position summary CSV
        try:
            if table_rows:
                pd.DataFrame(table_rows).to_csv(out_dir / "per_position_summary.csv", index=False)
        except Exception:
            pass
        if save_results:
            self._save_results(results, analysis_name)
        return results


def main():
    """CLI interface for running analyses."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BAN Authentication Analysis Framework")
    parser.add_argument("--data-dir", type=str, default="./data", 
                       help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Path to output directory (default: data_dir/results)")
    parser.add_argument("--analysis", type=str, default="all",
                       choices=[
                       "all",
                       "offbody-position",
                       "device",
                       "movement",
                       "mixed",
                       "hierarchical",
                       "verification",
                       "offbody-deviceid-per-position"
                       ],
                       help="Analysis to run")
    parser.add_argument("--max-samples", type=int, default=5000,
                       help="Maximum samples per class")
    parser.add_argument("--no-advanced-features", action="store_true",
                       help="Disable advanced feature extraction")
    parser.add_argument("--no-deep-learning", action="store_true",
                       help="Disable deep learning models")
    parser.add_argument("--balance-by", type=str, nargs="*", default=None,
                       help="Default balance_by keys (e.g., dvc pos_label session)")
    parser.add_argument("--source-path", type=str, default=None,
                       help="Path to specific source file (e.g., unified.pkl)")
    args = parser.parse_args()
    
    # Initialize framework
    framework = AnalysisFramework(
        data_root=args.data_dir,
        output_root=args.output_dir,
        max_samples_per_class=args.max_samples,
        use_advanced_features=not args.no_advanced_features,
        use_deep_learning=not args.no_deep_learning,
        default_balance_by=args.balance_by,
    )
    
    # Run specified analysis
    source_path = args.source_path
    
    if args.analysis == "all":
        results = framework.run_all_analyses(source_path)
        logger.info("Completed all analyses")
        logger.info(f"Summary: {json.dumps(results['summary'], indent=2)}")
    elif args.analysis == "offbody-position":
        results = framework.analyze_offbody_position(source_path)
        logger.info(f"Offbody position global accuracy: {(results.get('global', {}) or {}).get('holdout', {}).get('accuracy', 0):.4f}")
    elif args.analysis == "device":
        results = framework.analyze_device_identification("both", source_path)
        logger.info(f"OnBody device accuracy: {results.get('onBody', {}).get('holdout', {}).get('accuracy', 0):.4f}")
        logger.info(f"OffBody device accuracy: {results.get('offBody', {}).get('holdout', {}).get('accuracy', 0):.4f}")
    elif args.analysis == "movement":
        results = framework.analyze_onbody_movement(True, source_path)
        logger.info(f"Global movement accuracy: {results.get('global', {}).get('holdout', {}).get('accuracy', 0):.4f}")
    elif args.analysis == "mixed":
        results = framework.analyze_mixed_scenario(source_path)
        logger.info(f"Mixed scenario (baseline) accuracy: {results.get('baseline', {}).get('holdout', {}).get('accuracy', 0):.4f}")
    elif args.analysis == "hierarchical":
        results = framework.analyze_hierarchical_classification(source_path)
        logger.info(f"Hierarchical overall accuracy: {results.get('hierarchical_results', {}).get('overall_accuracy', 0):.4f}")
    elif args.analysis == "verification":
        results = framework.analyze_offbody_verification(source_path)
        logger.info(f"Verification macro AUC: {results.get('macro_auc', 0):.4f}, macro EER: {results.get('macro_eer', 1):.4f}")
    elif args.analysis == "offbody-deviceid-per-position":
        results = framework.analyze_offbody_device_id_per_position(source_path)
        logger.info(f"OffBody per-position device-ID macro accuracy: {results.get('macro_acc', 0):.4f}")
    
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()