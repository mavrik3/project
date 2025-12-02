import logging
from pathlib import Path
from analysis_framework import AnalysisFramework

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def _default_source_spec():
    """Return a source spec that includes both onBody and offBody dataset directories.
    This allows analyses that need both scenarios to work without relying on unified.pkl.
    """
    base = "C:/Users/msherazi/BAN Auth Project/data"
    return {
        "onBody": f"{base}/datasets/onbody_iqbb_hybrid",
        "offBody": f"{base}/datasets/offbody_iqbb_hybrid",
        # If you prefer to use the unified file (when available), uncomment below:
        # "unified": f"{base}/unified.pkl",
    }

def run_offbody_position_analysis():
    """Example of running offBody position analysis."""
    # Initialize the framework
    framework = AnalysisFramework(
        data_root="C:/Users/msherazi/BAN Auth Project/data",
        output_root="C:/Users/msherazi/BAN Auth Project/results09172025",
        max_samples_per_class=10000,
        use_advanced_features=True,
        use_deep_learning=True,
        # Surface balanced sampling across device/position/session
        default_balance_by=["dvc", "pos_label", "session"],
    )
    
    # Use unified or per-scenario spec; both are accepted
    source_spec = _default_source_spec()
    
    # Run offBody position analysis
    results = framework.analyze_offbody_position(
        source_path=source_spec
    )
    
    # Print results
    logger.info(f"OffBody position analysis completed")
    logger.info(f"Accuracy: {results.get('holdout', {}).get('accuracy', 0):.4f}")
    logger.info(f"F1 weighted: {results.get('holdout', {}).get('f1_weighted', 0):.4f}")
    
    # If deep learning was used
    if "deep_learning" in results:
        logger.info(f"Deep learning model: {results['deep_learning'].get('model_type', 'unknown')}")
        logger.info(f"Deep learning accuracy: {results['deep_learning'].get('best_accuracy', 0):.4f}")
    
    return results

def run_offbody_verification_analysis():
    """Run verification-style offBody position analysis (per-device ROC/AUC/EER)."""
    framework = AnalysisFramework(
        data_root="C:/Users/msherazi/BAN Auth Project/data",
        output_root="C:/Users/msherazi/BAN Auth Project/results",
        max_samples_per_class=10000,
        use_advanced_features=True,
        use_deep_learning=False,
        default_balance_by=["dvc", "pos_label", "session"],
    )
    source_spec = _default_source_spec()
    results = framework.analyze_offbody_verification(source_path=source_spec, min_samples_per_cell=20)
    if "error" not in results:
        logger.info("OffBody verification analysis completed")
        logger.info(f"Macro AUC: {results.get('macro_auc', 0):.4f}")
        logger.info(f"Macro EER: {results.get('macro_eer', 1):.4f}")
        logger.info(f"Devices evaluated: {len(results.get('per_device', {}))}")
    else:
        logger.error(results.get("error"))
    return results

def run_offbody_device_id_per_position_analysis():
    """Run per-position device identification on offBody samples."""
    framework = AnalysisFramework(
        data_root="C:/Users/msherazi/BAN Auth Project/data",
        output_root="C:/Users/msherazi/BAN Auth Project/results",
        max_samples_per_class=10000,
        use_advanced_features=True,
        use_deep_learning=True,
        default_balance_by=["dvc", "pos_label", "session"],
    )
    source_spec = _default_source_spec()
    results = framework.analyze_offbody_device_id_per_position(source_path=source_spec, min_samples_per_cell=20)
    if "error" not in results:
        logger.info("OffBody per-position device ID analysis completed")
        logger.info(f"Macro accuracy: {results.get('macro_acc', 0):.4f}")
        logger.info(f"Positions evaluated: {len(results.get('per_position', {}))}")
    else:
        logger.error(results.get("error"))
    return results

def run_device_identification_analysis():
    """Example of running device identification analysis."""
    framework = AnalysisFramework(
        data_root="C:/Users/msherazi/BAN Auth Project/data",
        output_root="C:/Users/msherazi/BAN Auth Project/results",
        max_samples_per_class=10000,
        use_advanced_features=True,
        default_balance_by=["dvc", "pos_label", "session"],
    )
    
    source_spec = _default_source_spec()
    
    # Run device identification for both scenarios
    results = framework.analyze_device_identification(
        scenario="both",
        source_path=source_spec
    )
    
    # Print results
    logger.info(f"Device identification analysis completed")
    logger.info(f"OnBody device accuracy: {results.get('onBody', {}).get('holdout', {}).get('accuracy', 0):.4f}")
    logger.info(f"OffBody device accuracy: {results.get('offBody', {}).get('holdout', {}).get('accuracy', 0):.4f}")
    
    return results

def run_onbody_movement_analysis(per_device=True):
    """
    Run analysis of onBody movement patterns.
    
    Args:
        per_device: Whether to analyze movement per device (default: True)
        
    Returns:
        Dictionary with analysis results
    """
    framework = AnalysisFramework(
        data_root="C:/Users/msherazi/BAN Auth Project/data",
        output_root="C:/Users/msherazi/BAN Auth Project/results",
        max_samples_per_class=10000,
        use_advanced_features=True,
        default_balance_by=["dvc", "pos_label", "session"],
    )
    
    source_spec = _default_source_spec()
    
    # Run onBody movement analysis
    results = framework.analyze_onbody_movement(
        per_device=per_device,
        source_path=source_spec
    )
    
    # Print results
    logger.info(f"OnBody movement analysis completed")
    
    # Global movement results
    if "global" in results:
        global_acc = results["global"].get("holdout", {}).get("accuracy", 0)
        global_f1 = results["global"].get("holdout", {}).get("f1_weighted", 0)
        logger.info(f"Global movement accuracy: {global_acc:.4f}")
        logger.info(f"Global movement F1 weighted: {global_f1:.4f}")
    
    # Per-device results if available
    if per_device and "per_device" in results:
        logger.info("Per-device movement classification results:")
        for device_id, device_results in results["per_device"].items():
            dev_acc = device_results.get("holdout", {}).get("accuracy", 0)
            logger.info(f"  {device_id}: accuracy = {dev_acc:.4f}")
    
    return results

def run_mixed_scenario_analysis():
    """
    Run analysis of mixed scenario classification (onBody vs offBody).
    
    Returns:
        Dictionary with analysis results
    """
    framework = AnalysisFramework(
        data_root="C:/Users/msherazi/BAN Auth Project/data",
        output_root="C:/Users/msherazi/BAN Auth Project/results",
        max_samples_per_class=10000,
        use_advanced_features=True,
        use_deep_learning=True,
        default_balance_by=["dvc", "pos_label", "session"],
    )
    
    source_spec = _default_source_spec()
    
    # Run mixed scenario analysis
    results = framework.analyze_mixed_scenario(
        source_path=source_spec
    )
    
    # Print results
    logger.info(f"Mixed scenario analysis completed")
    
    # Baseline results (with device fingerprints)
    baseline_acc = results.get("baseline", {}).get("holdout", {}).get("accuracy", 0)
    logger.info(f"Baseline accuracy (with device fingerprints): {baseline_acc:.4f}")
    
    # Robust results (without device fingerprints)
    robust_acc = results.get("robust", {}).get("holdout", {}).get("accuracy", 0)
    logger.info(f"Robust accuracy (without device fingerprints): {robust_acc:.4f}")
    
    # Leaky split results
    leaky_acc = results.get("leaky", {}).get("holdout", {}).get("accuracy", 0)
    logger.info(f"Leaky split accuracy (non-grouped): {leaky_acc:.4f}")
    
    return results

def run_hierarchical_classification_analysis():
    """
    Run hierarchical classification analysis:
    1. First classify scenario (onBody vs offBody)
    2a. If onBody: use device prediction as position
    2b. If offBody: classify device, then position
    
    Returns:
        Dictionary with analysis results
    """
    framework = AnalysisFramework(
        data_root="C:/Users/msherazi/BAN Auth Project/data",
        output_root="C:/Users/msherazi/BAN Auth Project/results",
        max_samples_per_class=50000,
        use_advanced_features=True,
        use_deep_learning=True,
        default_balance_by=["dvc", "pos_label", "session"],
    )
    
    source_spec = _default_source_spec()
    
    # Run hierarchical classification analysis
    results = framework.analyze_hierarchical_classification(
        source_path=source_spec
    )
    
    # Print results
    logger.info(f"Hierarchical classification analysis completed")
    
    # Extract hierarchical results
    h_results = results.get("hierarchical_results", {})
    
    # Print accuracies at each level
    logger.info(f"Scenario classification accuracy: {h_results.get('scenario_accuracy', 0):.4f}")
    logger.info(f"OnBody position accuracy: {h_results.get('onbody_accuracy', 0):.4f}")
    logger.info(f"OffBody device accuracy: {h_results.get('offbody_device_accuracy', 0):.4f}")
    logger.info(f"OffBody position accuracy: {h_results.get('offbody_position_accuracy', 0):.4f}")
    logger.info(f"Overall position accuracy: {h_results.get('overall_accuracy', 0):.4f}")
    
    return results

def run_comprehensive_analysis():
    """Run all analyses and get a comprehensive report."""
    framework = AnalysisFramework(
        data_root="C:/Users/msherazi/BAN Auth Project/data",
        output_root="C:/Users/msherazi/BAN Auth Project/results09202025",
        max_samples_per_class=50000,
        default_balance_by=["dvc", "pos_label", "session"],
    )
    
    source_spec = _default_source_spec()
    
    # Run all analyses
    results = framework.run_all_analyses(
        source_path=source_spec
    )
    
    # Print summary
    logger.info(f"All analyses completed")
    logger.info(f"Summary:")
    for key, value in results["summary"].items():
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    return results

if __name__ == "__main__":
    # Choose which analysis to run by uncommenting one of the following:
    
    # Single analysis options
    # analysis_results = run_offbody_position_analysis()
    # analysis_results = run_offbody_verification_analysis()
    # analysis_results = run_offbody_device_id_per_position_analysis()
    # analysis_results = run_device_identification_analysis()
    # analysis_results = run_onbody_movement_analysis(per_device=True)
    # analysis_results = run_mixed_scenario_analysis()
    analysis_results = run_hierarchical_classification_analysis()
    
    # Or run everything
    # analysis_results = run_comprehensive_analysis()
    
    logger.info(f"Results saved to {Path('C:/Users/msherazi/BAN Auth Project/results')}")