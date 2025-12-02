import logging
import time
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence, cast

import numpy as np
import pandas as pd
# Use a non-interactive backend for matplotlib to avoid Tkinter thread errors
import os as _os
_os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib as _mpl
try:
    _mpl.use("Agg", force=True)
except Exception:
    pass
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    VotingClassifier, IsolationForest
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, GroupKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# Add NearestCentroid for baseline models
from sklearn.neighbors import NearestCentroid

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Data Structures and Constants
# -----------------------------------------------------------------------------

class ModelType(Enum):
    """Model types for both ML and DL models."""
    # ML models
    RANDOM_FOREST = "rf"
    EXTRA_TREES = "et"
    GRADIENT_BOOSTING = "gb"
    SVM = "svc"
    NEAREST_CENTROID = "nc"
    LOGISTIC_REGRESSION = "lr"
    KNN = "knn"
    # DL models
    MLP = "mlp"
    CNN1D = "cnn1d"
    CNN_RNN = "cnn_rnn"
    
    @classmethod
    def ml_models(cls) -> List["ModelType"]:
        """Return all ML model types."""
        return [
            cls.RANDOM_FOREST,
            cls.SVM,
            cls.NEAREST_CENTROID,
            cls.EXTRA_TREES, cls.GRADIENT_BOOSTING,
            cls.LOGISTIC_REGRESSION, cls.KNN
        ]
    
    @classmethod
    def dl_models(cls) -> List["ModelType"]:
        """Return all DL model types."""
        return [cls.MLP, cls.CNN1D, cls.CNN_RNN]

@dataclass
class TrainingConfig:
    """Configuration for both ML and DL training."""
    # Common parameters
    task_name: str
    use_grouped_split: bool = True
    test_size: float = 0.2
    random_state: int = RANDOM_SEED
    # New: number of attempts for robust split search and optional forced models
    split_attempts: int = 15
    forced_models: Optional[List[str]] = None
    
    # ML specific parameters
    accuracy_threshold: float = 0.90
    n_jobs: int = -1
    
    # DL specific parameters
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0.0 < self.test_size < 1.0, "Test size must be between 0 and 1"
        assert self.epochs > 0 or self.epochs == 0, "Epochs must be non-negative"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.weight_decay >= 0, "Weight decay must be non-negative"
        
    @property
    def effective_batch_size(self) -> int:
        """Return batch size adjusted for dataset size."""
        return min(self.batch_size, 32)  # Default fallback

@dataclass
class TrainingResult:
    """Training results for both ML and DL models."""
    model_name: str
    accuracy: float
    f1_weighted: float
    predictions: np.ndarray
    model: Optional[Any] = None  # Either sklearn model or PyTorch model
    training_time: float = 0.0
    # Add the corresponding true targets for the predictions to enable correct confusion matrices
    targets: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "accuracy": float(self.accuracy),
            "f1_weighted": float(self.f1_weighted),
            "training_time": float(self.training_time)
        }

@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    task_name: str
    best_result: TrainingResult
    candidate_results: List[TrainingResult]
    holdout_metrics: Dict[str, Any]
    cross_val_metrics: Optional[Dict[str, Any]] = None
    deep_learning_result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.task_name,
            "model": self.best_result.model_name,
            "holdout": self.holdout_metrics,
            "candidates": [r.to_dict() for r in self.candidate_results],
            **({"logo": self.cross_val_metrics} if self.cross_val_metrics else {}),
            **({"deep_learning": self.deep_learning_result} if self.deep_learning_result else {})
        }

# -----------------------------------------------------------------------------
# Traditional ML Models
# -----------------------------------------------------------------------------

class MLModelRegistry:
    """Registry for traditional ML models."""
    
    @staticmethod
    def create_model(model_type: ModelType, random_state: int = RANDOM_SEED, n_jobs: int = -1) -> BaseEstimator:
        """Create a model based on the specified type."""
        if model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=300, 
                random_state=random_state, 
                n_jobs=n_jobs, 
                class_weight="balanced_subsample"
            )
        elif model_type == ModelType.EXTRA_TREES:
            return ExtraTreesClassifier(
                n_estimators=400, 
                random_state=random_state, 
                n_jobs=n_jobs, 
                class_weight="balanced_subsample"
            )
        elif model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(random_state=random_state)
        elif model_type == ModelType.SVM:
            return make_pipeline(
                StandardScaler(with_mean=True),
                SVC(C=3.0, gamma="scale", probability=True, class_weight="balanced", random_state=random_state)
            )
        elif model_type == ModelType.NEAREST_CENTROID:
            return make_pipeline(
                StandardScaler(with_mean=True),
                NearestCentroid()
            )
        elif model_type == ModelType.LOGISTIC_REGRESSION:
            return make_pipeline(
                StandardScaler(with_mean=True),
                LogisticRegression(max_iter=4000, solver="saga", class_weight="balanced", n_jobs=n_jobs)
            )
        elif model_type == ModelType.KNN:
            return make_pipeline(
                StandardScaler(with_mean=True),
                KNeighborsClassifier(n_neighbors=11, weights="distance")
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class MLTrainer:
    """Trainer for traditional ML models."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer."""
        self.config = config

    def _robust_train_test_split(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Find a leakage-safe split with class overlap and balanced distributions.
        Returns train_idx, test_idx, used_grouped.
        """
        n = len(y)
        rng = np.random.RandomState(self.config.random_state)
        y_arr = np.asarray(y)
        
        # Helper to check class coverage
        def ok_split(tr: np.ndarray, te: np.ndarray) -> bool:
            y_tr, y_te = y_arr[tr], y_arr[te]
            return (
                len(np.unique(y_tr)) >= 2 and len(np.unique(y_te)) >= 2 and
                set(np.unique(y_te)).issubset(set(np.unique(y_tr)))
            )
        
        # Divergence between train/test class distributions (lower is better)
        def div_score(tr: np.ndarray, te: np.ndarray) -> float:
            cls = np.unique(y_arr)
            # distributions
            p = np.array([np.mean(y_arr[tr] == c) for c in cls])
            q = np.array([np.mean(y_arr[te] == c) for c in cls])
            return float(0.5 * np.sum(np.abs(p - q)))  # L1/2 distance
        
        # Prefer grouped split when requested and groups available
        if self.config.use_grouped_split and groups is not None:
            g_arr = np.asarray(groups)
            best: Optional[Tuple[np.ndarray, np.ndarray]] = None
            best_score = float("inf")
            attempts = max(3, self.config.split_attempts)
            for a in range(attempts):
                try:
                    splitter = GroupShuffleSplit(n_splits=1, test_size=self.config.test_size, random_state=self.config.random_state + a)
                    tr, te = next(splitter.split(X, y_arr, g_arr))
                except ValueError:
                    break
                if not ok_split(tr, te):
                    continue
                # Ensure minimal per-class test coverage (>=2 where possible)
                y_te = y_arr[te]
                uniq, cnt = np.unique(y_te, return_counts=True)
                if np.any(cnt < 2) and len(te) >= 2 * len(uniq):
                    # try next attempt to avoid too-thin test classes
                    continue
                score = div_score(tr, te)
                if score < best_score:
                    best = (tr, te)
                    best_score = score
            if best is not None:
                return best[0], best[1], True
            # If no acceptable grouped split, fall back to stratified (leaky upper bound)
            logger.warning(f"[{self.config.task_name}] Could not find balanced grouped split after {attempts} attempts. Falling back to stratified split (non-grouped).")
        
        # Non-grouped stratified split
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.config.test_size, random_state=self.config.random_state)
            tr, te = next(sss.split(np.zeros(len(y_arr)), y_arr))
            # Ensure both sides have >=2 classes (rare edge)
            if not ok_split(tr, te):
                # Last resort: filter test to seen classes
                seen = set(np.unique(y_arr[tr]))
                te = np.array([i for i in te if y_arr[i] in seen])
            return tr, te, False
        except Exception:
            # Ultimate fallback: random split
            indices = np.arange(n)
            rng.shuffle(indices)
            split = int((1 - self.config.test_size) * n)
            tr, te = indices[:split], indices[split:]
            return tr, te, False
        
    def _compute_divergence(self, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> float:
        cls = np.unique(y)
        p = np.array([np.mean(y[tr] == c) for c in cls])
        q = np.array([np.mean(y[te] == c) for c in cls])
        return float(0.5 * np.sum(np.abs(p - q)))

    def _compute_ece(self, y_true: np.ndarray, proba: np.ndarray, class_labels: Sequence[Any], n_bins: int = 15) -> float:
        if proba is None or proba.size == 0:
            return float('nan')
        # Map y_true to indices of class_labels
        label_to_idx = {str(c): i for i, c in enumerate(class_labels)}
        y_idx = np.array([label_to_idx.get(str(v), -1) for v in y_true])
        mask = y_idx >= 0
        if not np.any(mask):
            return float('nan')
        confidences = np.max(proba[mask], axis=1)
        predictions = np.argmax(proba[mask], axis=1)
        correctness = (predictions == y_idx[mask]).astype(float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            m = (confidences >= bins[i]) & (confidences < bins[i + 1]) if i < n_bins - 1 else (confidences >= bins[i]) & (confidences <= bins[i + 1])
            if not np.any(m):
                continue
            acc_bin = float(np.mean(correctness[m]))
            conf_bin = float(np.mean(confidences[m]))
            ece += (np.sum(m) / len(confidences)) * abs(acc_bin - conf_bin)
        return float(ece)

    def _temp_scale_probs(self, proba: np.ndarray, T: float) -> np.ndarray:
        # Temperature scaling directly on probabilities (approximation)
        p = np.clip(proba, 1e-12, 1.0)
        p_t = p ** (1.0 / max(1e-3, T))
        p_t = p_t / p_t.sum(axis=1, keepdims=True)
        return p_t

    def _separability_metrics(self, X: np.ndarray, y: np.ndarray, tr: np.ndarray) -> Dict[str, float]:
        # Project to 1D via PCA on train set and compute pairwise d' and Bhattacharyya on PC1
        try:
            from sklearn.decomposition import PCA
            X_tr = X[tr]
            y_tr = y[tr]
            if X_tr.ndim != 2 or len(np.unique(y_tr)) < 2:
                return {}
            pca = PCA(n_components=1, random_state=self.config.random_state)
            z = pca.fit_transform(X_tr).ravel()
            vals = {}
            classes = np.unique(y_tr)
            means = {c: float(np.mean(z[y_tr == c])) for c in classes}
            vars_ = {c: float(np.var(z[y_tr == c]) + 1e-9) for c in classes}
            dprimes = []
            bhs = []
            for i, ci in enumerate(classes):
                for cj in classes[i + 1:]:
                    mu_i, mu_j = means[ci], means[cj]
                    s_i, s_j = np.sqrt(vars_[ci]), np.sqrt(vars_[cj])
                    dprime = abs(mu_i - mu_j) / np.sqrt(0.5 * (s_i ** 2 + s_j ** 2))
                    # Bhattacharyya distance for 1D Gaussians
                    bh = 0.25 * ((mu_i - mu_j) ** 2) / (s_i ** 2 + s_j ** 2) + 0.5 * np.log((s_i ** 2 + s_j ** 2) / (2 * s_i * s_j))
                    dprimes.append(float(dprime))
                    bhs.append(float(bh))
            if not dprimes or not bhs:
                return {}
            vals["mean_pairwise_dprime_pc1"] = float(np.mean(dprimes))
            vals["mean_pairwise_bhattacharyya_pc1"] = float(np.mean(bhs))
            return vals
        except Exception:
            return {}

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, model_type: ModelType, 
                           train_idx: np.ndarray, test_idx: np.ndarray) -> TrainingResult:
        """Train and evaluate a single model."""
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create and train the model
        model = MLModelRegistry.create_model(
            model_type, 
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        except Exception as e:
            logger.error(f"Error training {model_type.value}: {e}")
            predictions = np.zeros_like(y_test)
            accuracy = 0.0
            f1 = 0.0
            model = None
        training_time = time.time() - start_time
        
        return TrainingResult(
            model_name=model_type.value,
            accuracy=accuracy,
            f1_weighted=f1,
            predictions=predictions,
            model=model,
            training_time=training_time,
            targets=y_test
        )
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, top_models: List[TrainingResult],
                       train_idx: np.ndarray, test_idx: np.ndarray, max_models: int = 3) -> Optional[TrainingResult]:
        """Train an ensemble model from the top performing models."""
        # Filter models that support probabilities
        prob_capable_models = []
        for result in top_models[:max_models]:
            if result.model is None or result.accuracy == 0:
                continue
                
            model_type = ModelType(result.model_name)
            if model_type in [ModelType.RANDOM_FOREST, 
                            #   ModelType.EXTRA_TREES, 
                            #   ModelType.GRADIENT_BOOSTING, ModelType.SVM, 
                            #   ModelType.LOGISTIC_REGRESSION
                            ]:
                prob_capable_models.append(result.model_name)
            
            if len(prob_capable_models) >= max_models:
                break
        
        if len(prob_capable_models) < 2:
            return None  # Not enough models for ensemble
        
        # Create estimators for ensemble
        estimators = []
        for model_name in prob_capable_models:
            model = MLModelRegistry.create_model(
                ModelType(model_name),
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            estimators.append((model_name, model))
        
        # Create and train ensemble
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        try:
            # Try with n_jobs if available
            ensemble = VotingClassifier(
                estimators=estimators, 
                voting="soft", 
                weights=[1.0] * len(estimators),
                n_jobs=self.config.n_jobs
            )
        except TypeError:
            # Fallback without n_jobs for older sklearn versions
            ensemble = VotingClassifier(
                estimators=estimators, 
                voting="soft", 
                weights=[1.0] * len(estimators)
            )
        
        start_time = time.time()
        try:
            ensemble.fit(X_train, y_train)
            predictions = ensemble.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            ensemble_name = f"voting({' + '.join(prob_capable_models)})"
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return None
        training_time = time.time() - start_time
        
        return TrainingResult(
            model_name=ensemble_name,
            accuracy=accuracy,
            f1_weighted=f1,
            predictions=predictions,
            model=ensemble,
            training_time=training_time,
            targets=y_test
        )
    
    def perform_leave_one_group_out_cv(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                                      best_model_type: ModelType) -> Dict[str, Any]:
        """Perform leave-one-group-out cross-validation."""
        logo = LeaveOneGroupOut()
        accuracies = []
        total_groups = int(len(np.unique(groups))) if groups is not None and len(groups) == len(y) else None
        skipped_folds = 0
        evaluated_folds = 0
        
        for train_idx, test_idx in logo.split(X, y, groups):
            # Skip if train set doesn't contain all classes
            if len(np.unique(y[train_idx])) < len(np.unique(y)):
                skipped_folds += 1
                continue
                
            # Train a new model of the same type as the best model
            try:
                model = MLModelRegistry.create_model(
                    best_model_type,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
                model.fit(X[train_idx], y[train_idx])
                predictions = model.predict(X[test_idx])
                acc = accuracy_score(y[test_idx], predictions)
                accuracies.append(acc)
                evaluated_folds += 1
            except Exception as e:
                logger.warning(f"Error in LOGO CV fold: {e}")
                skipped_folds += 1
                continue
        
        # Return cross-validation metrics
        return {
            "folds": evaluated_folds,
            "accuracy_mean": None if not accuracies else float(np.mean(accuracies)),
            "accuracy_std": None if not accuracies else float(np.std(accuracies)),
            "total_groups": total_groups,
            "folds_skipped": skipped_folds,
        }

    def train_and_select_best(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> EvaluationResult:
        """Train multiple models and select the best one."""
        # Compute robust split first
        train_idx, test_idx, used_grouped = self._robust_train_test_split(X, y, groups)
        
        # Start with RF for backward compatibility
        results = [self.train_and_evaluate(X, y, ModelType.RANDOM_FOREST, train_idx, test_idx)]
        best_result = results[0]
        
        # Always evaluate any user-forced baseline models (e.g., svc, nc)
        forced = []
        if self.config.forced_models:
            for name in self.config.forced_models:
                try:
                    mt = ModelType(name) if isinstance(name, str) else ModelType(name.value)
                except Exception:
                    continue
                if mt == ModelType.RANDOM_FOREST:
                    continue  # already included
                res = self.train_and_evaluate(X, y, mt, train_idx, test_idx)
                results.append(res)
                forced.append(mt.value)
                if res.accuracy > best_result.accuracy:
                    best_result = res
        
        # If accuracy is below threshold, try other models
        if best_result.accuracy < self.config.accuracy_threshold:
            for model_type in [mt for mt in ModelType.ml_models() if mt != ModelType.RANDOM_FOREST and mt.value not in forced]:
                result = self.train_and_evaluate(X, y, model_type, train_idx, test_idx)
                results.append(result)
                if result.accuracy > best_result.accuracy:
                    best_result = result
                if best_result.accuracy >= self.config.accuracy_threshold:
                    break
            # Try ensemble as a last resort
            if best_result.accuracy < self.config.accuracy_threshold:
                sorted_results = sorted(results, key=lambda r: r.accuracy, reverse=True)
                ensemble_result = self.train_ensemble(X, y, sorted_results, train_idx, test_idx)
                if ensemble_result and ensemble_result.accuracy > best_result.accuracy:
                    best_result = ensemble_result
                    results.append(ensemble_result)
        
        # Compute holdout metrics with diagnostics
        holdout_metrics = {
            "accuracy": best_result.accuracy,
            "f1_weighted": best_result.f1_weighted,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_classes": int(len(np.unique(y))),
            "used_grouped_split": bool(used_grouped),
            "classes_in_train": sorted(list(map(str, np.unique(y[train_idx])))),
            "classes_in_test": sorted(list(map(str, np.unique(y[test_idx])))),
            # New diagnostics
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "divergence_score": self._compute_divergence(np.asarray(y), train_idx, test_idx),
        }
        
        # Add ECE if probabilities available
        try:
            model = best_result.model
            if hasattr(model, "predict_proba"):
                # Attempt to get underlying classifier in a pipeline
                clf = getattr(model, "steps", None)
                if clf:
                    classifier = clf[-1][1]
                    proba = model.predict_proba(X[test_idx])
                    class_labels = getattr(classifier, "classes_", np.unique(y))
                else:
                    proba = model.predict_proba(X[test_idx])
                    class_labels = getattr(model, "classes_", np.unique(y))
                ece = self._compute_ece(best_result.targets, proba, class_labels)
                # Temperature scaling (approximate on probs)
                best_ece = ece
                best_T = 1.0
                for T in (0.7, 0.85, 1.0, 1.2, 1.5):
                    proba_t = self._temp_scale_probs(proba, T)
                    ece_t = self._compute_ece(best_result.targets, proba_t, class_labels)
                    if np.isfinite(ece_t) and ece_t < best_ece:
                        best_ece = ece_t
                        best_T = T
                holdout_metrics["ece_uncalibrated"] = float(ece) if np.isfinite(ece) else None
                holdout_metrics["ece_temp_scaled_best"] = float(best_ece) if np.isfinite(best_ece) else None
                holdout_metrics["ece_temp_scaled_best_T"] = float(best_T) if np.isfinite(best_ece) else None
                
                # Binary ROC/AUC/EER and threshold selection when applicable
                try:
                    if class_labels is not None and len(class_labels) == 2 and best_result.targets is not None:
                        from sklearn.metrics import roc_curve, auc
                        # Map true labels to indices
                        label_to_idx = {str(c): i for i, c in enumerate(class_labels)}
                        y_idx = np.array([label_to_idx.get(str(v), -1) for v in best_result.targets])
                        mask = y_idx >= 0
                        if np.any(mask):
                            fpr, tpr, thr = roc_curve(y_idx[mask], proba[mask, 1])
                            auc_val = float(auc(fpr, tpr))
                            holdout_metrics["roc_auc"] = auc_val
                            # EER
                            fnr = 1.0 - tpr
                            e_idx = int(np.argmin(np.abs(fpr - fnr)))
                            eer = float((fpr[e_idx] + fnr[e_idx]) / 2.0)
                            tau_eer = float(thr[e_idx]) if e_idx < len(thr) else None
                            holdout_metrics["eer"] = eer
                            holdout_metrics["tau_eer"] = tau_eer
                            # Thresholds at target FPRs
                            def _tau_at_target_fpr(target: float) -> Optional[float]:
                                if fpr.size == 0:
                                    return None
                                idx = int(np.argmin(np.abs(fpr - target)))
                                return float(thr[idx]) if idx < len(thr) else None
                            holdout_metrics["tau_fpr_1pct"] = _tau_at_target_fpr(0.01)
                            holdout_metrics["tau_fpr_5pct"] = _tau_at_target_fpr(0.05)
                except Exception:
                    pass
        except Exception:
            pass
        
        # Separability diagnostics on train split
        sep = self._separability_metrics(X, y, train_idx)
        if sep:
            holdout_metrics["separability"] = sep
        
        # Perform cross-validation if grouped was used
        cv_metrics = None
        if used_grouped and groups is not None:
            try:
                base_name = best_result.model_name.split("(")[0]
                # Robustly map base_name to ModelType if possible
                try:
                    best_model_type = ModelType(base_name)
                except Exception:
                    best_model_type = ModelType.RANDOM_FOREST
            except Exception:
                best_model_type = ModelType.RANDOM_FOREST
            try:
                cv_metrics = self.perform_leave_one_group_out_cv(X, y, groups, best_model_type)
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
        
        return EvaluationResult(
            task_name=self.config.task_name,
            best_result=best_result,
            candidate_results=results,
            holdout_metrics=holdout_metrics,
            cross_val_metrics=cv_metrics
        )

# -----------------------------------------------------------------------------
# Deep Learning Models
# -----------------------------------------------------------------------------

class FeatureDataset(Dataset):
    """PyTorch dataset for feature-based classification."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, class_names: Optional[List[str]] = None):
        self.features = features.astype(np.float32)
        self.labels = np.array([str(v) for v in labels], dtype=object)
        
        # Create categorical mapping for labels (shared if class_names provided)
        if class_names is not None and len(class_names) > 0:
            label_series = pd.Categorical(self.labels, categories=class_names, ordered=True)
        else:
            label_series = pd.Categorical(self.labels)
        self.label_codes = label_series.codes
        self.class_names = list(label_series.categories)
        self.num_classes = len(self.class_names)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        feature = torch.from_numpy(self.features[idx]).float()
        label = torch.tensor(self.label_codes[idx], dtype=torch.long)
        return feature, label

class IQDataset(Dataset):
    """PyTorch dataset for raw IQ data."""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, class_names: Optional[List[str]] = None):
        # Handle complex input and convert to appropriate format
        if np.iscomplexobj(data):
            self.data = np.stack([data.real, data.imag], axis=-1).astype(np.float32)
        elif data.ndim == 2:  # (n_samples, n_features)
            # Reshape for 1D convolution (B, T, C) -> (B, C, T)
            self.data = np.expand_dims(data, axis=-1).transpose(0, 2, 1).astype(np.float32)
        else:
            self.data = data.astype(np.float32)
            
            # Ensure format is (B, C, T) for CNN models
            if self.data.ndim == 3 and self.data.shape[2] == 2:  # (B, T, 2)
                self.data = self.data.transpose(0, 2, 1)  # -> (B, 2, T)
        
        self.labels = np.array([str(v) for v in labels], dtype=object)
        
        # Create categorical mapping for labels (shared if class_names provided)
        if class_names is not None and len(class_names) > 0:
            label_series = pd.Categorical(self.labels, categories=class_names, ordered=True)
        else:
            label_series = pd.Categorical(self.labels)
        self.label_codes = label_series.codes
        self.class_names = list(label_series.categories)
        self.num_classes = len(self.class_names)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        data = torch.from_numpy(self.data[idx]).float()
        label = torch.tensor(self.label_codes[idx], dtype=torch.long)
        return data, label

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, 
                stride: int = 1, padding: Optional[int] = None, activation: str = "gelu"):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
            
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class MLPClassifier(nn.Module):
    """MLP classifier for extracted features."""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class CNN1D(nn.Module):
    """1D CNN for IQ signal classification."""
    
    def __init__(self, input_channels: int = 2, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        
        self.backbone = nn.Sequential(
            ConvBlock(input_channels, 32, 9),
            nn.MaxPool1d(2),
            ConvBlock(32, 64, 7),
            nn.MaxPool1d(2),
            ConvBlock(64, 128, 5),
            nn.MaxPool1d(2),
            ConvBlock(128, 128, 5)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class CNNBiRNNClassifier(nn.Module):
    """CNN + Bidirectional RNN classifier with residual connections."""
    
    def __init__(self, rnn_type: str = "lstm", input_channels: int = 2,
                cnn_features: int = 128, hidden_size: int = 192,
                num_layers: int = 2, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            ConvBlock(input_channels, 32, 9),
            nn.MaxPool1d(2),
            ConvBlock(32, 64, 7),
            nn.MaxPool1d(2),
            ConvBlock(64, cnn_features, 5)
        )
        
        # Bidirectional RNN
        rnn_class = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_class(
            input_size=cnn_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Global pooling and residual projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.residual_proj = nn.Linear(cnn_features, 2 * hidden_size)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, num_classes)
        )
    
    def forward(self, x):
        # CNN feature extraction: (B, 2, T) -> (B, C, T')
        cnn_features = self.cnn(x)
        
        # Prepare for RNN: (B, C, T') -> (B, T', C)
        rnn_input = cnn_features.transpose(1, 2)
        
        # Bidirectional RNN: (B, T', C) -> (B, T', 2H)
        rnn_output, _ = self.rnn(rnn_input)
        
        # Temporal averaging of RNN output
        rnn_pooled = rnn_output.mean(dim=1)  # (B, 2H)
        
        # Global pooling of CNN features for residual
        cnn_pooled = self.global_pool(cnn_features).squeeze(-1)  # (B, C)
        residual = self.residual_proj(cnn_pooled)  # (B, 2H)
        
        # Combine RNN and CNN features
        combined = rnn_pooled + residual
        
        return self.classifier(combined)

class DLModelRegistry:
    """Registry for deep learning models."""
    
    @staticmethod
    def create_model(model_type: ModelType, input_dim: int, num_classes: int) -> nn.Module:
        """Create a model based on the specified type."""
        if model_type == ModelType.MLP:
            return MLPClassifier(input_dim, num_classes)
        elif model_type == ModelType.CNN1D:
            # For CNN1D, input_dim is interpreted as the number of channels
            channels = 1 if input_dim == 1 else 2  # Default to 2 channels for I/Q data
            return CNN1D(input_channels=channels, num_classes=num_classes)
        elif model_type == ModelType.CNN_RNN:
            # For CNN+RNN, input_dim is interpreted as the number of channels
            channels = 1 if input_dim == 1 else 2
            return CNNBiRNNClassifier(
                rnn_type="lstm",
                input_channels=channels,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown DL model type: {model_type}")

class DLTrainer:
    """Trainer for deep learning models."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def create_data_loaders(self, X: np.ndarray, y: np.ndarray, 
                          train_idx: np.ndarray, test_idx: np.ndarray,
                          model_type: ModelType,
                          class_names: Optional[List[str]] = None) -> Tuple[DataLoader, DataLoader, List[str]]:
        """Create appropriate data loaders based on model type and data format."""
        
        # Determine dataset type based on model type
        if model_type == ModelType.MLP:
            # For MLP, use feature dataset
            train_dataset = FeatureDataset(X[train_idx], y[train_idx], class_names=class_names)
            test_dataset = FeatureDataset(X[test_idx], y[test_idx], class_names=class_names)
        else:
            # For CNN/RNN models, use IQ dataset
            train_dataset = IQDataset(X[train_idx], y[train_idx], class_names=class_names)
            test_dataset = IQDataset(X[test_idx], y[test_idx], class_names=class_names)
        
        # Calculate effective batch size based on dataset size
        train_batch_size = min(self.config.batch_size, max(8, len(train_dataset) // 4))
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=len(train_dataset) >= train_batch_size,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=min(self.config.batch_size, len(test_dataset)),
            shuffle=False,
            pin_memory=True
        )
        
        # Use provided class_names if given, else adopt from dataset
        final_class_names = class_names if class_names is not None else train_dataset.class_names
        return train_loader, test_loader, list(final_class_names)
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch and return average loss."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / max(1, num_batches)
    
    def evaluate(self, model: nn.Module, test_loader: DataLoader, criterion: nn.Module) -> Dict[str, Any]:
        """Evaluate model on validation/test data."""
        model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                predictions = logits.argmax(dim=1).cpu().numpy()
                targets = batch_y.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                total_loss += loss.item()
                num_batches += 1
        
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        avg_loss = total_loss / max(1, num_batches)
        
        return {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "loss": avg_loss,
            "predictions": np.array(all_predictions),
            "targets": np.array(all_targets)
        }
    
    def determine_input_dimension(self, X: np.ndarray, model_type: ModelType) -> int:
        """Determine input dimension based on data and model type."""
        if model_type == ModelType.MLP:
            return X.shape[1] if X.ndim == 2 else X.size // len(X)
        else:
            # For CNN models, input_dim is the number of channels
            if np.iscomplexobj(X):
                return 2  # Complex data becomes 2 channels (real, imag)
            elif X.ndim == 3 and X.shape[2] == 2:
                return 2  # Already in (B, T, 2) format
            elif X.ndim == 3 and X.shape[1] == 2:
                return 2  # Already in (B, 2, T) format
            else:
                return 1  # Single channel
    
    def train(self, X: np.ndarray, y: np.ndarray, model_type: ModelType, 
             train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
        """Train a deep learning model and return results."""
        try:
            # Determine and fix class mapping across train/test
            # y is already converted to strings upstream
            all_classes = sorted(list(map(str, np.unique(y))))
            num_classes = len(all_classes)
            
            # Create data loaders with shared class mapping
            train_loader, test_loader, class_names = self.create_data_loaders(
                X, y, train_idx, test_idx, model_type, class_names=all_classes
            )
            
            # Determine input dimension
            input_dim = self.determine_input_dimension(X, model_type)
            
            # Create model
            model = DLModelRegistry.create_model(model_type, input_dim, num_classes)
            model.to(self.device)
            
            # Setup training
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            criterion = nn.CrossEntropyLoss()
            
            # Setup learning rate scheduler
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
            
            # Training loop
            best_accuracy = 0.0
            best_metrics = None
            best_epoch = 0
            patience_counter = 0
            
            start_time = time.time()
            
            for epoch in range(self.config.epochs):
                # Train
                train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
                
                # Evaluate
                test_metrics = self.evaluate(model, test_loader, criterion)
                
                # Update scheduler
                scheduler.step()
                
                # Track best model
                if test_metrics["accuracy"] > best_accuracy:
                    best_accuracy = test_metrics["accuracy"]
                    best_metrics = test_metrics.copy()
                    best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                # Log progress
                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == self.config.epochs - 1:
                    logger.info(
                        f"[DL-{model_type.value.upper()}] Epoch {epoch+1}/{self.config.epochs}: "
                        f"train_loss={train_loss:.4f}, test_acc={test_metrics['accuracy']:.4f}"
                    )
            
            training_time = time.time() - start_time
            
            # Return results
            return {
                "model_type": model_type.value,
                "best_accuracy": float(best_accuracy),
                "best_f1_weighted": float(best_metrics["f1_weighted"]),
                "best_epoch": int(best_epoch),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_classes": num_classes,
                "class_names": class_names,
                "training_time": training_time
            }
            
        except Exception as e:
            logger.exception(f"Error training neural network with model type {model_type}")
            return {
                "error": str(e), 
                "model_type": model_type.value,
                "best_accuracy": 0.0,
                "best_f1_weighted": 0.0
            }

    def train_best_model(self, X: np.ndarray, y: np.ndarray, 
                        train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
        """Train the most suitable model based on data characteristics."""
        # Auto-detect best model type based on input data characteristics
        if X.ndim == 2:  # Regular feature matrix
            model_type = ModelType.MLP
        elif X.ndim == 3 or np.iscomplexobj(X):  # IQ data
            # Choose between CNN and CNN+RNN based on sequence length
            if np.iscomplexobj(X):
                avg_seq_len = X.shape[1] if X.ndim > 1 else X.shape[0]
            else:
                avg_seq_len = X.shape[1] if X.shape[1] > X.shape[2] else X.shape[2]
            
            model_type = ModelType.CNN_RNN if avg_seq_len > 1000 else ModelType.CNN1D
        else:
            model_type = ModelType.MLP  # Default to MLP for any other case
        
        logger.info(f"[DL] Auto-selected model type: {model_type.value}")
        
        # Train the selected model
        return self.train(X, y, model_type, train_idx, test_idx)

# -----------------------------------------------------------------------------
# Unified API for ML/DL Training
# -----------------------------------------------------------------------------

class UnifiedTrainer:
    """Unified API for both ML and DL model training."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.ml_trainer = MLTrainer(config)
        self.dl_trainer = DLTrainer(config)
    
    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> EvaluationResult:
        """Train both ML and DL models, returning a comprehensive evaluation."""
        # Validate inputs
        if X.size == 0 or len(np.unique(y)) < 2:
            logger.warning(f"[{self.config.task_name}] Insufficient data or classes")
            return EvaluationResult(
                task_name=self.config.task_name,
                best_result=TrainingResult(
                    model_name="none", 
                    accuracy=0.0, 
                    f1_weighted=0.0, 
                    predictions=np.array([]),
                    model=None,
                    targets=np.array([])
                ),
                candidate_results=[],
                holdout_metrics={"skipped": True}
            )
        
        # Convert labels to strings for consistency
        y = np.array([str(v) for v in y], dtype=object)
        
        # Train ML models
        logger.info(f"[{self.config.task_name}] Training ML models...")
        ml_result = self.ml_trainer.train_and_select_best(X, y, groups)
        
        # Train DL models if enough data
        dl_result = None
        if len(X) >= 200 and (torch.cuda.is_available() or len(X) < 10000):
            logger.info(f"[{self.config.task_name}] Training DL model...")
            try:
                # Use a robust split similar to ML for DL
                tr_idx, te_idx, _ = self.ml_trainer._robust_train_test_split(X, y, groups if self.config.use_grouped_split else None)
                dl_result = self.dl_trainer.train_best_model(X, y, tr_idx, te_idx)
                if "error" not in dl_result:
                    logger.info(
                        f"[{self.config.task_name}] DL model: {dl_result['model_type']}, "
                        f"accuracy: {dl_result['best_accuracy']:.4f}"
                    )
            except Exception as e:
                logger.warning(f"[{self.config.task_name}] DL training failed: {e}")
        
        # Add DL result to ML evaluation
        ml_result.deep_learning_result = dl_result
        
        return ml_result

# -----------------------------------------------------------------------------
# Helper Functions for Visualization
# -----------------------------------------------------------------------------

def generate_visualizations(result: EvaluationResult, 
                           X: np.ndarray, 
                           y: np.ndarray, 
                           feature_names: Optional[List[str]],
                           out_dir: Path) -> Dict[str, str]:
    """Generate visualizations for the evaluation result."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    visualizations = {}
    
    # Convenience handles
    best_model = result.best_result.model
    y_true = getattr(result.best_result, "targets", None)
    hold = result.holdout_metrics or {}
    test_idx = np.array(hold.get("test_indices", []), dtype=int) if isinstance(hold.get("test_indices", []), list) else np.array([])
    
    # Generate feature importance plot if the model supports it
    if best_model is not None:
        try:
            # Extract feature importances
            if hasattr(best_model, "feature_importances_"):
                importances = best_model.feature_importances_
            elif hasattr(best_model, "coef_"):
                importances = np.abs(best_model.coef_).mean(axis=0) if best_model.coef_.ndim > 1 else np.abs(best_model.coef_)
            elif hasattr(best_model, "steps") and hasattr(best_model.steps[-1][1], "coef_"):
                importances = np.abs(best_model.steps[-1][1].coef_).mean(axis=0) if best_model.steps[-1][1].coef_.ndim > 1 else np.abs(best_model.steps[-1][1].coef_)
            else:
                importances = None
                
            if importances is not None:
                plt.figure(figsize=(10, 6))
                # Get top features
                top_k = min(25, len(importances))
                indices = np.argsort(importances)[-top_k:]
                
                plt.barh(range(top_k), importances[indices])
                feature_labels = [
                    feature_names[i] if feature_names and i < len(feature_names) else f"Feature {i}" 
                    for i in indices
                ]
                plt.yticks(range(top_k), feature_labels)
                plt.xlabel("Importance")
                plt.title(f"Top {top_k} Feature Importances")
                plt.tight_layout()
                
                fi_path = out_dir / f"{result.task_name}_feature_importance.png"
                plt.savefig(fi_path, dpi=150)
                plt.close()
                
                visualizations["feature_importance"] = str(fi_path)
        except Exception as e:
            logger.warning(f"Feature importance visualization failed: {e}")
    
    # Reliability diagram (ECE) if probabilities are available
    try:
        if best_model is not None and y_true is not None and test_idx.size > 0 and hasattr(best_model, "predict_proba"):
            # Obtain probabilities for test set
            proba = best_model.predict_proba(X[test_idx])
            classes = None
            if hasattr(best_model, "steps"):
                classes = best_model.steps[-1][1].classes_
            elif hasattr(best_model, "classes_"):
                classes = best_model.classes_
            # Use max confidence and correctness for multiclass reliability
            y_true_str = np.array([str(v) for v in y_true], dtype=object)
            # Map to indices; if mapping fails, skip reliability
            if classes is not None:
                label_to_idx = {str(c): i for i, c in enumerate(classes)}
                y_idx = np.array([label_to_idx.get(str(v), -1) for v in y_true_str])
                mask = (y_idx >= 0)
                if np.any(mask):
                    confidences = np.max(proba[mask], axis=1)
                    preds = np.argmax(proba[mask], axis=1)
                    correctness = (preds == y_idx[mask]).astype(float)
                    bins = np.linspace(0.0, 1.0, 16)
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    accs = []
                    confs = []
                    counts = []
                    for i in range(len(bins) - 1):
                        m = (confidences >= bins[i]) & (confidences < bins[i + 1] if i < len(bins) - 2 else confidences <= bins[i + 1])
                        if not np.any(m):
                            accs.append(np.nan)
                            confs.append(np.nan)
                            counts.append(0)
                        else:
                            accs.append(float(np.mean(correctness[m])))
                            confs.append(float(np.mean(confidences[m])))
                            counts.append(int(np.sum(m)))
                    # Plot
                    plt.figure(figsize=(6, 5))
                    plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
                    plt.plot([c for c in confs if np.isfinite(c)], [a for a in accs if np.isfinite(a)], marker='o', label='Empirical')
                    plt.xlabel('Confidence')
                    plt.ylabel('Accuracy')
                    plt.title(f"Reliability: {result.task_name}")
                    plt.legend(loc='best')
                    plt.tight_layout()
                    rel_path = out_dir / f"{result.task_name}_reliability.png"
                    plt.savefig(rel_path, dpi=150)
                    plt.close()
                    visualizations["reliability"] = str(rel_path)
                    # Save CSV
                    rel_csv = out_dir / f"{result.task_name}_reliability_bins.csv"
                    df_rel = pd.DataFrame({
                        "bin_lower": bins[:-1],
                        "bin_upper": bins[1:],
                        "avg_conf": confs,
                        "acc": accs,
                        "count": counts
                    })
                    df_rel.to_csv(rel_csv, index=False)
                    visualizations["reliability_bins_csv"] = str(rel_csv)
    except Exception as e:
        logger.warning(f"Reliability plotting failed: {e}")
    
    # Generate PCA plot
    try:
        if len(X) > 3 and X.ndim == 2:  # Only for tabular data
            # Use a subset for large datasets
            if len(X) > 3000:
                indices = np.random.choice(len(X), 3000, replace=False)
                X_subset = X[indices]
                y_subset = y[indices]
            else:
                X_subset = X
                y_subset = y
                
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_subset)
            
            plt.figure(figsize=(8, 6))
            
            # Get unique classes for PCA visualization
            pca_classes = np.unique(y_subset)
            
            # Define color palette
            n_colors = len(pca_classes)
            if n_colors <= 10:
                palette = sns.color_palette("tab10", n_colors)
            else:
                palette = sns.color_palette("husl", n_colors)
            
            # Plot each class
            for i, cls in enumerate(pca_classes):
                mask = y_subset == cls
                plt.scatter(
                    X_pca[mask, 0], X_pca[mask, 1], 
                    s=30, alpha=0.7, 
                    label=cls,
                    color=palette[i % len(palette)]
                )
            
            plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%})")
            plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%})")
            plt.title(f"PCA Visualization: {result.task_name}")
            plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            pca_path = out_dir / f"{result.task_name}_pca.png"
            plt.savefig(pca_path, dpi=150)
            plt.close()
            
            visualizations["pca"] = str(pca_path)
            
            # Try UMAP as an additional visualization if available
            try:
                import umap  # type: ignore
                reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
                X_umap = reducer.fit_transform(X_subset)
                
                plt.figure(figsize=(8, 6))
                for i, cls in enumerate(pca_classes):
                    mask = y_subset == cls
                    plt.scatter(
                        X_umap[mask, 0], X_umap[mask, 1],
                        s=30, alpha=0.7,
                        label=cls,
                        color=palette[i % len(palette)]
                    )
                plt.title(f"UMAP Visualization: {result.task_name}")
                plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                umap_path = out_dir / f"{result.task_name}_umap.png"
                plt.savefig(umap_path, dpi=150)
                plt.close()
                visualizations["umap"] = str(umap_path)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"PCA visualization failed: {e}")
    
    # Generate confusion matrix (and CSV)
    try:
        best_result = result.best_result
        y_true_cm = getattr(best_result, "targets", None)
        y_pred = best_result.predictions if best_result is not None else None
        
        if y_true_cm is not None and y_pred is not None and len(y_true_cm) == len(y_pred) and len(y_true_cm) > 0:
            # Ensure both are numpy arrays of strings for consistent labeling
            y_true_cm = np.array([str(v) for v in y_true_cm], dtype=object)
            y_pred = np.array([str(v) for v in y_pred], dtype=object)
            
            # Derive consistent, deterministic class order
            cm_classes = np.unique(np.concatenate([y_true_cm, y_pred]))
            cm_classes = np.sort(cm_classes)
            
            # Build matrix using explicit labels so axes line up with ticks
            cm = confusion_matrix(y_true_cm, y_pred, labels=cm_classes)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, annot=(cm.shape[0] <= 10), fmt="d", cmap="Blues", 
                xticklabels=cm_classes, yticklabels=cm_classes
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix: {result.task_name}")
            plt.tight_layout()
            
            cm_path = out_dir / f"{result.task_name}_confusion_matrix.png"
            plt.savefig(cm_path, dpi=150)
            plt.close()
            visualizations["confusion_matrix"] = str(cm_path)
            # Save numeric CM as CSV
            try:
                cm_csv = out_dir / f"{result.task_name}_confusion_matrix.csv"
                df_cm = pd.DataFrame(cm, index=cm_classes, columns=cm_classes)
                df_cm.to_csv(cm_csv)
                visualizations["confusion_matrix_csv"] = str(cm_csv)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Confusion matrix visualization failed: {e}")
    
    # ROC curve for binary tasks
    try:
        if best_model is not None and y_true is not None and test_idx.size > 0 and hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(X[test_idx])
            classes = None
            if hasattr(best_model, "steps"):
                classes = best_model.steps[-1][1].classes_
            elif hasattr(best_model, "classes_"):
                classes = best_model.classes_
            if classes is not None and len(classes) == 2:
                # Map y_true to 0/1
                cls_to_idx = {str(c): i for i, c in enumerate(classes)}
                y_idx = np.array([cls_to_idx.get(str(v), -1) for v in y_true])
                mask = y_idx >= 0
                if np.any(mask):
                    fpr, tpr, thr = roc_curve(y_idx[mask], proba[mask, 1])
                    auc_val = float(auc(fpr, tpr))
                    plt.figure(figsize=(6, 5))
                    plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlabel("FPR")
                    plt.ylabel("TPR")
                    plt.title(f"ROC: {result.task_name}")
                    plt.legend(loc="lower right")
                    plt.tight_layout()
                    roc_path = out_dir / f"{result.task_name}_roc.png"
                    plt.savefig(roc_path, dpi=150)
                    plt.close()
                    visualizations["roc_curve"] = str(roc_path)
                    # CSV of ROC points
                    roc_csv = out_dir / f"{result.task_name}_roc.csv"
                    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(roc_csv, index=False)
                    visualizations["roc_csv"] = str(roc_csv)
    except Exception as e:
        logger.warning(f"ROC plotting failed: {e}")
    
    return visualizations

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def device_position_matrix(df: pd.DataFrame) -> Dict:
    if df.empty or "dvc" not in df.columns or "pos_label" not in df.columns:
        return {}
    pivot = df.pivot_table(index="dvc", columns="pos_label", values="frame_idx",
                           aggfunc="count", fill_value=0)
    per_dev_positions = (pivot > 0).sum(axis=1)
    per_pos_devices = (pivot > 0).sum(axis=0)
    one_to_one_devices = int((per_dev_positions == 1).sum())
    one_to_one_positions = int((per_pos_devices == 1).sum())
    perfect_confound = (
        one_to_one_devices == len(per_dev_positions)
        and one_to_one_positions == len(per_pos_devices)
        and len(per_dev_positions) == len(per_pos_devices)
    )
    if perfect_confound:
        logging.warning("[diagnostic] Perfect device ↔ position confound detected.")
    return {
        "n_devices": int(per_dev_positions.shape[0]),
        "n_positions": int(per_pos_devices.shape[0]),
        "devices_single_position": one_to_one_devices,
        "positions_single_device": one_to_one_positions,
        "perfect_confounded": perfect_confound
    }

def eval_anomaly_per_class(X: np.ndarray, y: np.ndarray) -> Dict:
    if X.size == 0:
        return {}
    out = {}
    for lab in np.unique(y):
        mask = y == lab
        if mask.sum() < 30:
            continue
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        iso.fit(X[mask])
        sc = -iso.decision_function(X[mask])
        out[str(lab)] = {
            "mean_score": float(np.mean(sc)),
            "std_score": float(np.std(sc)),
            "n": int(mask.sum())
        }
    return out