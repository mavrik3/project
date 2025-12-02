from ml_dl_framework import (
    TrainingConfig, UnifiedTrainer, EvaluationResult, 
    generate_visualizations
)
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import logging
import json
import time

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

from sklearn.metrics import roc_auc_score

def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    task_name: str,
    output_dir: Optional[Union[str, Path]] = None,
    feature_names: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    use_grouped_split: bool = True,
    remove_fingerprint: bool = False,
    manifest_extra: Optional[Dict[str, Any]] = None,
    kept_frame_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate both ML and DL models on the given dataset.
    
    Args:
        X: Feature matrix (n_samples, n_features) or IQ data
        y: Target labels (n_samples,)
        groups: Group labels for grouped cross-validation (n_samples,)
        task_name: Name of the classification task
        output_dir: Directory for saving visualizations and results
        feature_names: Names of features (optional)
        config: Additional configuration parameters
        use_grouped_split: Whether to use grouped splits for evaluation
        remove_fingerprint: Deprecated. Fingerprint removal is now applied pre-feature at the IQ stage.
            This flag is kept for backward compatibility and is ignored.
        manifest_extra: Optional metadata to merge into saved/returned results (e.g., badges, class_counts).
            Will be added at the top level of the results dictionary.
            Example: {"badges": {...}, "class_counts": {...}}
        kept_frame_indices: Optional mapping from local sample index to global frame_id (aligned with X order)
            Used to persist exact train/test frame IDs for dataset cards and split diagnostics.
            
    Returns:
        Dictionary with evaluation results
    """
    # Prepare configuration with default forced baselines
    cfg = {**(config or {})}
    if "forced_models" not in cfg:
        cfg["forced_models"] = ["svc", "nc"]  # always evaluate SVM and Nearest Centroid baselines
    
    training_config = TrainingConfig(
        task_name=task_name,
        use_grouped_split=use_grouped_split,
        **cfg
    )
    
    # Prepare output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Deprecated path: do not modify post-engineered features here
    if remove_fingerprint:
        logging.info(f"[{task_name}] remove_fingerprint requested. Skipping: fingerprint removal is performed before feature engineering now.")
    
    # Train models
    trainer = UnifiedTrainer(training_config)
    evaluation = trainer.train(X, y, groups)
    
    # Build base dict
    evaluation_dict = evaluation.to_dict()
    
    # Add run id
    run_id = f"{task_name}_{int(time.time())}"
    evaluation_dict["result_run_id"] = run_id
    
    # Persist best model artifact if possible
    model_path = None
    try:
        if output_dir and joblib is not None and evaluation.best_result.model is not None:
            model_path = output_dir / f"{task_name}_best_model.joblib"
            joblib.dump(evaluation.best_result.model, model_path)
            evaluation_dict["model_path"] = str(model_path)
    except Exception as e:
        logging.warning(f"[{task_name}] Failed to persist trained model: {e}")
    
    # Generate visualizations if output directory is provided
    if output_dir:
        visualizations = generate_visualizations(
            evaluation, X, y, feature_names, output_dir
        )
        evaluation_dict["visualizations"] = visualizations
        # Merge manifest metadata if provided
        if manifest_extra:
            try:
                evaluation_dict.update(manifest_extra)
            except Exception:
                logging.warning("Failed to merge manifest_extra into results; ignoring metadata.")
        
        # Ensure chance baseline is present
        try:
            hold = evaluation_dict.get("holdout", {})
            if "chance_baseline" not in evaluation_dict:
                k = max(1, int(hold.get("n_classes", 0) or (len(np.unique(y)) if len(y) > 0 else 1)))
                evaluation_dict["chance_baseline"] = 1.0 / float(k)
        except Exception:
            pass
        
        # Save table-ready summary CSV (single-row)
        try:
            # Extract key metrics
            hold = evaluation_dict.get("holdout", {})
            logo = evaluation_dict.get("logo", {})
            row: Dict[str, Any] = {
                "result_run_id": run_id,
                "task": task_name,
                "model": evaluation_dict.get("model"),
                "used_grouped_split": hold.get("used_grouped_split"),
                "n_train": hold.get("n_train"),
                "n_test": hold.get("n_test"),
                "n_classes": hold.get("n_classes"),
                "accuracy": hold.get("accuracy"),
                "f1_weighted": hold.get("f1_weighted"),
                "ece_uncalibrated": hold.get("ece_uncalibrated"),
                "ece_temp_scaled_best": hold.get("ece_temp_scaled_best"),
                "ece_temp_scaled_best_T": hold.get("ece_temp_scaled_best_T"),
                # Binary metrics if available
                "roc_auc": hold.get("roc_auc"),
                "eer": hold.get("eer"),
                "tau_eer": hold.get("tau_eer"),
                "tau_fpr_1pct": hold.get("tau_fpr_1pct"),
                "tau_fpr_5pct": hold.get("tau_fpr_5pct"),
                # Also emit FAR alias keys for compatibility
                "tau_far_1pct": hold.get("tau_fpr_1pct"),
                "tau_far_5pct": hold.get("tau_fpr_5pct"),
                # CV diagnostics
                "logo_accuracy_mean": logo.get("accuracy_mean"),
                "logo_accuracy_std": logo.get("accuracy_std"),
                "logo_folds": logo.get("folds"),
                "logo_total_groups": logo.get("total_groups"),
                "logo_folds_skipped": logo.get("folds_skipped"),
                "chance_baseline": evaluation_dict.get("chance_baseline"),
            }
            # If roc_auc wasn't computed upstream, try to compute it here as a fallback
            if row.get("roc_auc") is None:
                try:
                    best_model = evaluation.best_result.model
                    test_idx = np.array(hold.get("test_indices", []), dtype=int)
                    y_true = evaluation.best_result.targets
                    if best_model is not None and hasattr(best_model, "predict_proba") and y_true is not None and test_idx.size > 0:
                        proba = best_model.predict_proba(X[test_idx])
                        classes = None
                        if hasattr(best_model, "steps"):
                            classes = best_model.steps[-1][1].classes_
                        elif hasattr(best_model, "classes_"):
                            classes = best_model.classes_
                        if classes is not None and len(classes) == 2:
                            cls_to_idx = {str(c): i for i, c in enumerate(classes)}
                            y_idx = np.array([cls_to_idx.get(str(v), -1) for v in y_true])
                            mask = y_idx >= 0
                            if np.any(mask):
                                auc_val = roc_auc_score(y_idx[mask], proba[mask, 1])
                                row["roc_auc"] = float(auc_val)
                except Exception:
                    pass
            # Include badges if present
            badges = evaluation_dict.get("badges", {}) or {}
            row.update({f"badge_{k}": v for k, v in badges.items()})
            # Write CSV
            summary_path = Path(output_dir) / f"{task_name}_summary.csv"
            pd.DataFrame([row]).to_csv(summary_path, index=False)
        except Exception as e:
            logging.warning(f"[{task_name}] Failed to write table summary CSV: {e}")
        
        # Results data pack directory
        try:
            pack_dir = Path(output_dir) / f"{task_name}__{run_id}"
            pack_dir.mkdir(parents=True, exist_ok=True)
            # Split diagnostics: map local indices to global frame ids if available
            hold = evaluation_dict.get("holdout", {})
            local_train = hold.get("train_indices", []) or []
            local_test = hold.get("test_indices", []) or []
            train_frame_ids: Optional[List[int]] = None
            test_frame_ids: Optional[List[int]] = None
            if kept_frame_indices and isinstance(kept_frame_indices, (list, tuple)):
                try:
                    train_frame_ids = [int(kept_frame_indices[i]) for i in local_train if i < len(kept_frame_indices)]
                    test_frame_ids = [int(kept_frame_indices[i]) for i in local_test if i < len(kept_frame_indices)]
                    evaluation_dict["train_frame_ids"] = train_frame_ids
                    evaluation_dict["test_frame_ids"] = test_frame_ids
                except Exception:
                    pass
            # Persist calibration sweep (ECE vs T) if probabilities available
            try:
                best_model = evaluation.best_result.model
                y_true = evaluation.best_result.targets
                test_idx = np.array(local_test, dtype=int)
                calib_csv = pack_dir / "calibration_sweep.csv"
                if best_model is not None and hasattr(best_model, "predict_proba") and y_true is not None and test_idx.size > 0:
                    proba = best_model.predict_proba(X[test_idx])
                    classes = None
                    if hasattr(best_model, "steps"):
                        classes = best_model.steps[-1][1].classes_
                    elif hasattr(best_model, "classes_"):
                        classes = best_model.classes_
                    if classes is not None:
                        # sweep Ts
                        Ts = [0.6, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
                        # simple ECE util (duplicate of ml trainer approach)
                        def _ece(y_true_arr, proba_arr, class_labels, n_bins: int = 15) -> float:
                            label_to_idx = {str(c): i for i, c in enumerate(class_labels)}
                            y_idx = np.array([label_to_idx.get(str(v), -1) for v in y_true_arr])
                            mask = y_idx >= 0
                            if not np.any(mask):
                                return float("nan")
                            confidences = np.max(proba_arr[mask], axis=1)
                            predictions = np.argmax(proba_arr[mask], axis=1)
                            correctness = (predictions == y_idx[mask]).astype(float)
                            bins = np.linspace(0.0, 1.0, n_bins + 1)
                            ece_val = 0.0
                            for i in range(n_bins):
                                m = (confidences >= bins[i]) & (confidences < bins[i + 1] if i < n_bins - 1 else confidences <= bins[i + 1])
                                if not np.any(m):
                                    continue
                                acc_bin = float(np.mean(correctness[m]))
                                conf_bin = float(np.mean(confidences[m]))
                                ece_val += (np.sum(m) / len(confidences)) * abs(acc_bin - conf_bin)
                            return float(ece_val)
                        rows = []
                        for T in Ts:
                            p = np.clip(proba, 1e-12, 1.0)
                            p_t = p ** (1.0 / max(1e-3, T))
                            p_t = p_t / p_t.sum(axis=1, keepdims=True)
                            e = _ece(y_true, p_t, classes)
                            rows.append({"T": T, "ece": e})
                        pd.DataFrame(rows).to_csv(calib_csv, index=False)
                        evaluation_dict.setdefault("calibration", {})["sweep_csv"] = str(calib_csv)
            except Exception:
                pass
            # Persist separability table if present
            try:
                sep = (evaluation_dict.get("holdout", {}) or {}).get("separability", {})
                if sep:
                    sep_csv = pack_dir / "separability.csv"
                    pd.DataFrame([{**sep, "result_run_id": run_id}]).to_csv(sep_csv, index=False)
                    evaluation_dict.setdefault("separability", {})["csv"] = str(sep_csv)
            except Exception:
                pass
            # Persist split diagnostics CSV: per-class train/test counts and divergence
            try:
                hold = evaluation_dict.get("holdout", {})
                local_train = np.array(hold.get("train_indices", []) or [], dtype=int)
                local_test = np.array(hold.get("test_indices", []) or [], dtype=int)
                if local_train.size > 0 and local_test.size > 0 and len(y) == (local_train.size + local_test.size):
                    y_arr = np.array([str(v) for v in y], dtype=object)
                    classes = np.sort(np.unique(y_arr))
                    rows = []
                    n_train = local_train.size
                    n_test = local_test.size
                    for cls in classes:
                        m_tr = y_arr[local_train] == cls
                        m_te = y_arr[local_test] == cls
                        tr_cnt = int(np.sum(m_tr))
                        te_cnt = int(np.sum(m_te))
                        rows.append({
                            "class": str(cls),
                            "train_count": tr_cnt,
                            "test_count": te_cnt,
                            "train_frac": (tr_cnt / max(1, n_train)),
                            "test_frac": (te_cnt / max(1, n_test))
                        })
                    diag_csv = pack_dir / "split_diagnostics.csv"
                    pd.DataFrame(rows).to_csv(diag_csv, index=False)
                    evaluation_dict["split_diagnostics_csv"] = str(diag_csv)
            except Exception:
                pass
            # Create manifest with pointers to artifacts
            manifest = {
                "result_run_id": run_id,
                "task": task_name,
                "model": evaluation_dict.get("model"),
                "badges": evaluation_dict.get("badges", {}),
                "class_counts": evaluation_dict.get("class_counts", {}),
                "holdout": evaluation_dict.get("holdout", {}),
                "logo": evaluation_dict.get("logo", {}),
                "chance_baseline": evaluation_dict.get("chance_baseline"),
                "visualizations": evaluation_dict.get("visualizations", {}),
                "model_path": evaluation_dict.get("model_path"),
                "train_indices": local_train.tolist() if isinstance(local_train, np.ndarray) else (local_train or []),
                "test_indices": local_test.tolist() if isinstance(local_test, np.ndarray) else (local_test or []),
                "train_frame_ids": train_frame_ids,
                "test_frame_ids": test_frame_ids,
                "calibration_sweep_csv": str(calib_csv) if 'calib_csv' in locals() and isinstance(calib_csv, Path) and calib_csv.exists() else None,
                # expose binary thresholds if available
                "binary_thresholds": {
                    "tau_eer": (hold or {}).get("tau_eer"),
                    "tau_fpr_1pct": (hold or {}).get("tau_fpr_1pct"),
                    "tau_fpr_5pct": (hold or {}).get("tau_fpr_5pct"),
                    # Also include FAR aliases for compatibility
                    "tau_far_1pct": (hold or {}).get("tau_fpr_1pct"),
                    "tau_far_5pct": (hold or {}).get("tau_fpr_5pct"),
                },
                "split_diagnostics_csv": evaluation_dict.get("split_diagnostics_csv")
            }
            with open(pack_dir / "data_pack.json", "w") as f:
                json.dump(manifest, f, indent=2)
            evaluation_dict["data_pack_dir"] = str(pack_dir)
        except Exception as e:
            logging.warning(f"[{task_name}] Failed to build results data pack: {e}")
        
        # Save results to JSON
        result_path = output_dir / f"{task_name}_results.json"
        with open(result_path, "w") as f:
            json.dump(evaluation_dict, f, indent=2)
        
        logging.info(f"[{task_name}] Results saved to {result_path}")
        return evaluation_dict
    else:
        # Return dict without saving
        if manifest_extra:
            try:
                evaluation_dict.update(manifest_extra)
            except Exception:
                logging.warning("Failed to merge manifest_extra into results; ignoring metadata.")
        return evaluation_dict

def classify_with_visuals(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    out_dir: Path,
    grouped: bool = True,
    feature_names: Optional[List[str]] = None,
    acc_thresh: float = 0.90
) -> Dict[str, Any]:
    """
    Legacy-compatible function for the existing codebase.
    """
    config = {
        "accuracy_threshold": acc_thresh
    }
    
    return train_classifier(
        X=X,
        y=y,
        groups=g,
        task_name=name,
        output_dir=out_dir,
        feature_names=feature_names,
        config=config,
        use_grouped_split=grouped
    )

# Removed duplicate remove_device_fingerprint implementation to avoid divergence