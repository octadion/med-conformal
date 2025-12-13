import torch
import yaml
import numpy as np
import random
from pathlib import Path
import sys
import os

sys.path.append('.')
sys.path.append('/mnt/user-data/uploads')

from data.dataloader import MedMNISTDataLoader
from models.base_model import get_model, save_checkpoint
from models.conformal_wrapper import create_conformal_models
from models.gradcam import generate_gradcam_samples
from training.trainer import Trainer
from training.evaluator import ComprehensiveEvaluator
from visualization.plots import Visualizer
from lib_utils.logger import ExperimentLogger
import pandas as pd


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results_to_csv(all_results: dict, save_dir: Path):
    """Save results to CSV tables."""
    tables_dir = save_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)
    
    clf_data = []
    for method, results in all_results.items():
        clf = results['classification']
        clf_data.append({
            'Method': method,
            'Accuracy': clf['accuracy'],
            'Precision': clf['precision'],
            'Recall': clf['recall'],
            'F1-Score': clf['f1_score'],
            'AUC (Macro)': clf.get('auc_macro', 0.0)
        })
    
    df_clf = pd.DataFrame(clf_data)
    df_clf.to_csv(tables_dir / 'classification_metrics.csv', index=False)
    
    unc_data = []
    for method, results in all_results.items():
        if 'uncertainty' in results:
            unc = results['uncertainty']
            target = results['target_coverage']
            unc_data.append({
                'Method': method,
                'Coverage': unc['coverage'],
                'Target Coverage': target,
                'Coverage Violation': abs(unc['coverage'] - target),
                'Avg Set Size': unc['avg_set_size'],
                'Median Set Size': unc['efficiency']['median_size'],
                'Singleton Rate': unc['efficiency']['singleton_rate']
            })
    
    if unc_data:
        df_unc = pd.DataFrame(unc_data)
        df_unc.to_csv(tables_dir / 'uncertainty_metrics.csv', index=False)
    
    timing_data = []
    for method, results in all_results.items():
        timing = results['timing']
        timing_data.append({
            'Method': method,
            'Total Time (s)': timing['total_inference_time'],
            'Per Sample (ms)': timing['per_sample_time'] * 1000,
            'Throughput (img/s)': timing['throughput']
        })
    
    df_timing = pd.DataFrame(timing_data)
    df_timing.to_csv(tables_dir / 'timing_comparison.csv', index=False)
    
    print(f"\n✓ Results tables saved to: {tables_dir}")


def main():
    print("Stage 1: Setup & Configuration")
    print("-" * 60)
    
    config_path = Path('configs/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['experiment']['seed'])
    print(f"✓ Random seed set: {config['experiment']['seed']}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ No GPU available, using CPU")
    
    save_dir = Path(config['experiment']['save_dir'])
    logger = ExperimentLogger(
        log_dir=str(save_dir),
        experiment_name=config['experiment']['name']
    )
    logger.save_config(config)
    
    results_dir = logger.get_results_dir()
    print(f"✓ Results will be saved to: {results_dir}")

    print("\n" + "="*80)
    print("Stage 2: Data Loading")
    print("-" * 60)
    
    data_loader = MedMNISTDataLoader(config)
    train_loader, val_loader, test_loader = data_loader.load_data()
    
    class_names = data_loader.class_names
    num_classes = data_loader.num_classes
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  Classes: {class_names}")
    
    print("\n" + "="*80)
    print("Stage 3: Model Training")
    print("-" * 60)
    
    model = get_model(config, device)
    
    trainer = Trainer(model, config, device)
    
    trained_model = trainer.train(train_loader, val_loader, logger)
    
    logger.info(f"\n✓ Training completed!")
    logger.info(f"  Best validation accuracy: {trainer.best_val_acc:.4f}")
    
    print("\n" + "="*80)
    print("Stage 4: Conformal Prediction Setup")
    print("-" * 60)
    
    val_dataset = val_loader.dataset
    pct_paramtune = config['conformal']['pct_paramtune']
    paramtune_dataset, calib_dataset = data_loader.split_validation_set(
        val_dataset, pct_paramtune
    )
    
    batch_size = config['data']['batch_size']
    paramtune_loader = torch.utils.data.DataLoader(
        paramtune_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4
    )
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4
    )
    
    conformal_models = create_conformal_models(
        trained_model, paramtune_loader, calib_loader, config
    )
    
    logger.info(f"✓ Created {len(conformal_models)} conformal prediction models")
    
    print("\n" + "="*80)
    print("Stage 5: Evaluation")
    print("-" * 60)
    
    evaluator = ComprehensiveEvaluator(config, class_names)
    all_results = {}
    
    for method_name, conformal_model in conformal_models.items():
        is_main_method = (method_name == 'RAPS_Adaptive')
        
        results = evaluator.evaluate_method(
            conformal_model,
            test_loader,
            method_name,
            compute_gradcam=is_main_method
        )
        
        all_results[method_name] = results
        logger.log_test_results(method_name, results)
    
    print("\n" + "="*80)
    print("Stage 6: Methods Comparison")
    print("-" * 60)
    
    comparison = evaluator.compare_methods(all_results)
    evaluator.print_comparison(comparison)
    
    print("\n" + "="*80)
    print("Stage 7: Generating Visualizations")
    print("-" * 60)
    
    viz_dir = results_dir / 'visualizations'
    visualizer = Visualizer(viz_dir, config)
    
    train_history = logger.metrics_history['train']
    val_history = logger.metrics_history['val']
    
    visualizer.create_all_plots(
        train_history,
        val_history,
        all_results,
        class_names
    )
    
    if config['visualization']['generate']['gradcam_visualizations']:
        print("\n" + "="*80)
        print("Stage 8: Grad-CAM Visualization (Main Method)")
        print("-" * 60)
        
        gradcam_dir = viz_dir / 'gradcam'
        main_method = conformal_models['RAPS_Adaptive']
        
        generate_gradcam_samples(
            trained_model,
            main_method,
            test_loader,
            gradcam_dir,
            config,
            class_names
        )
    
    print("\n" + "="*80)
    print("Stage 9: Saving Results")
    print("-" * 60)
    
    logger.save_metrics()
    
    save_results_to_csv(all_results, results_dir)

    logger.create_summary()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nAll results saved to: {results_dir}")
    print("\nGenerated outputs:")
    print("  ✓ Training logs")
    print("  ✓ Model checkpoints")
    print("  ✓ Evaluation metrics (JSON & CSV)")
    print("  ✓ Comparison tables")
    print("  ✓ Training curves")
    print("  ✓ Confusion matrices")
    print("  ✓ Coverage analysis plots")
    print("  ✓ Set size distributions")
    print("  ✓ Grad-CAM visualizations")
    print("  ✓ Summary report")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Check summary: " + str(results_dir / f"{logger.experiment_name}_SUMMARY.txt"))
    print("  2. View plots: " + str(viz_dir))
    print("  3. Review tables: " + str(results_dir / 'tables'))
    print("  4. Use for manuscript!")
    print("="*80 + "\n")
    
    print("Best Performing Methods:")
    print("-" * 60)
    for metric, info in comparison['best'].items():
        print(f"  {metric.replace('_', ' ').title()}: {info['method']} ({info['value']:.4f})")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
