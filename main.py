import torch
import yaml
import numpy as np
import random
from pathlib import Path
import sys
import pandas as pd
import copy
import os

DRIVE_PATH = Path('/content/drive/MyDrive')

if DRIVE_PATH.exists():
    print(f"✅ Google Drive detected at {DRIVE_PATH}")
    BASE_OUTPUT_DIR = DRIVE_PATH / 'MedConformal_Final_Submission'
else:
    print("⚠️ Google Drive NOT detected. Using local 'outputs' folder.")
    BASE_OUTPUT_DIR = Path('outputs')

BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"📂 Output Directory: {BASE_OUTPUT_DIR}")

sys.path.append('.')

from data.dataloader import MedMNISTDataLoader
from models.base_model import get_model
from models.conformal_wrapper import StandardLAC, StandardAPS, SizeOptimizedRAPS, EntropyStratifiedRAPS, MondrianCP
from models.gradcam import generate_gradcam_samples
from training.trainer import Trainer
from training.evaluator import ComprehensiveEvaluator
from lib_utils.logger import ExperimentLogger

DATASETS_TO_RUN = ['organamnist', 'pathmnist']
SEEDS = [42, 10, 2024, 99, 123]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_paper_results(dataset_name, agg_results, save_dir):
    df = pd.DataFrame(agg_results)

    summary = df.groupby('Method').agg(['mean', 'std'])
    
    final_table = pd.DataFrame()
    metrics = ['Coverage', 'Avg_Set_Size', 'Hard_Case_Coverage']
    
    for metric in metrics:
        if metric in summary.columns:
            final_table[metric] = summary[metric].apply(
                lambda x: f"{x['mean']:.4f} ({x['std']:.4f})", axis=1
            )
    
    csv_path = save_dir / f'PAPER_TABLE_{dataset_name}.csv'
    final_table.to_csv(csv_path)
    
    df.to_csv(save_dir / f'RAW_RESULTS_{dataset_name}.csv', index=False)
    
    print(f"\n[RESULTS] Final Table for {dataset_name} saved to:")
    print(f"   -> {csv_path}")
    print(final_table)

def main():
    config_path = Path('configs/config.yaml')
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for dataset_name in DATASETS_TO_RUN:
        print(f"\n\n{'#'*60}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'#'*60}")

        current_config = copy.deepcopy(base_config)
        current_config['data']['dataset'] = dataset_name

        current_config['experiment']['save_dir'] = str(BASE_OUTPUT_DIR / f"{dataset_name}_final")
        
        save_dir = Path(current_config['experiment']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger = ExperimentLogger(str(save_dir), f"{dataset_name}_exp")

        set_seed(SEEDS[0]) 
        data_loader = MedMNISTDataLoader(current_config)
        train_loader, val_loader, test_loader = data_loader.load_data()
        
        print(f"\n[TRAINING] Training Base Model for {dataset_name}...")
        model = get_model(current_config, device)
        trainer = Trainer(model, current_config, device)
        trained_model = trainer.train(train_loader, val_loader, logger)
        
        dataset_results = []
        print(f"\n[EVALUATION] Starting Multi-Seed Conformal Evaluation...")
        
        # 2. Loop Seeds (Robustness)
        for seed_idx, seed in enumerate(SEEDS):
            print(f"   > Running Seed {seed}...")
            set_seed(seed)
            
            # Split Validation Random per Seed
            val_ds = val_loader.dataset
            paramtune_ds, calib_ds = data_loader.split_validation_set(
                val_ds, pct_paramtune=current_config['conformal']['pct_paramtune']
            )
            
            pt_loader = torch.utils.data.DataLoader(paramtune_ds, batch_size=32, shuffle=False)
            cal_loader = torch.utils.data.DataLoader(calib_ds, batch_size=32, shuffle=False)
            
            # 1. LAC
            lac = StandardLAC(trained_model, cal_loader, alpha=0.1, device=device)
            # 2. APS (Added Baseline)
            aps = StandardAPS(trained_model, cal_loader, alpha=0.1, device=device)
            # 3. RAPS Standard (Size Optimized)
            raps_std = SizeOptimizedRAPS(trained_model, cal_loader, pt_loader, alpha=0.1, k_reg=2, device=device)
            # 4. Entropy RAPS (Ours - Safety Optimized)
            ours = EntropyStratifiedRAPS(trained_model, cal_loader, pt_loader, alpha=0.1, k_reg=2, device=device)
            mondrian = MondrianCP(trained_model, cal_loader, pt_loader, alpha=0.1, n_strata=3, device=device)
            # Evaluate
            evaluator = ComprehensiveEvaluator(current_config, data_loader.class_names)

            res_lac = evaluator.evaluate_method(lac, test_loader, "LAC")
            res_aps = evaluator.evaluate_method(aps, test_loader, "APS")
            res_raps = evaluator.evaluate_method(raps_std, test_loader, "RAPS_Standard")
            res_ours = evaluator.evaluate_method(ours, test_loader, "EntropyRAPS", compute_gradcam=False)
            res_mondrian = evaluator.evaluate_method(mondrian, test_loader, "Mondrian")

            if seed_idx == 0 and dataset_name == 'organamnist':
                print("   [VISUALIZATION] Generating Grad-CAM Contrastive Maps...")
                viz_dir = save_dir / 'visualizations'
                generate_gradcam_samples(trained_model, ours, test_loader, viz_dir, current_config, data_loader.class_names)

            def get_hard_cov(res):
                if 'entropy_stratified' in res:
                    return res['entropy_stratified'].get('Hard (High Ent)', {}).get('coverage', res['uncertainty']['coverage'])
                return res['uncertainty']['coverage']

            dataset_results.append({
                'Seed': seed, 'Method': 'LAC',
                'Coverage': res_lac['uncertainty']['coverage'],
                'Avg_Set_Size': res_lac['uncertainty']['avg_set_size'],
                'Hard_Case_Coverage': get_hard_cov(res_lac)
            })
            dataset_results.append({
                'Seed': seed, 'Method': 'APS',
                'Coverage': res_aps['uncertainty']['coverage'],
                'Avg_Set_Size': res_aps['uncertainty']['avg_set_size'],
                'Hard_Case_Coverage': get_hard_cov(res_aps)
            })
            dataset_results.append({
                'Seed': seed, 'Method': 'RAPS_Standard',
                'Coverage': res_raps['uncertainty']['coverage'],
                'Avg_Set_Size': res_raps['uncertainty']['avg_set_size'],
                'Hard_Case_Coverage': get_hard_cov(res_raps)
            })
            dataset_results.append({
                'Seed': seed, 'Method': 'EntropyRAPS',
                'Coverage': res_ours['uncertainty']['coverage'],
                'Avg_Set_Size': res_ours['uncertainty']['avg_set_size'],
                'Hard_Case_Coverage': get_hard_cov(res_ours)
            })
            dataset_results.append({
                'Seed': seed, 'Method': 'Mondrian',
                'Coverage': res_mondrian['uncertainty']['coverage'],
                'Avg_Set_Size': res_mondrian['uncertainty']['avg_set_size'],
                'Hard_Case_Coverage': get_hard_cov(res_mondrian)
            })
            
        save_paper_results(dataset_name, dataset_results, save_dir)

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED.")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()