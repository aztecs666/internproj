"""
Master Training & Evaluation Pipeline - Walk-Forward + % Change Target
=======================================================================
Runs the complete ML pipeline:
1. Feature engineering (regenerate dataset with new features + % change target)
2. Train all Random Forest models (20 combos x 8 folds = 160 models)
3. Train all XGBoost models (20 combos x 8 folds = 160 models)
4. Evaluate all models on walk-forward test folds
5. Generate visualizations
"""

import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def run_pipeline():
    print("=" * 80)
    print("MASTER ML PIPELINE - Walk-Forward + % Change Target")
    print("=" * 80)
    start = time.time()

    # Step 0: Feature Engineering
    print(f"\n{'#' * 80}")
    print("# STEP 0: FEATURE ENGINEERING (regenerate dataset)")
    print(f"{'# ' * 40}")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "feature_engineering", os.path.join(BASE_DIR, "feature_engineering.py")
    )
    fe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe_mod)
    fe_mod.engineer_features(
        r"d:\Internproj\new dataset\final_ml_dataset.csv",
        r"d:\Internproj\new dataset\engineered_ml_dataset.csv",
    )

    # Step 1: Train Random Forest
    print(f"\n{'#' * 80}")
    print("# STEP 1: TRAIN RANDOM FOREST MODELS (Walk-Forward)")
    print(f"{'#' * 80}")
    rf_dir = os.path.join(BASE_DIR, "Random Forest")
    sys.path.insert(0, rf_dir)
    spec = importlib.util.spec_from_file_location(
        "train_rf", os.path.join(rf_dir, "train_rf.py")
    )
    train_rf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_rf_mod)
    train_rf_mod.train_all_rf_models()

    # Step 2: Train XGBoost
    print(f"\n{'#' * 80}")
    print("# STEP 2: TRAIN XGBOOST MODELS (Walk-Forward)")
    print(f"{'#' * 80}")
    xgb_dir = os.path.join(BASE_DIR, "XGBoost")
    sys.path.insert(0, xgb_dir)
    spec = importlib.util.spec_from_file_location(
        "train_xgb", os.path.join(xgb_dir, "train_xgb.py")
    )
    train_xgb_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_xgb_mod)
    train_xgb_mod.train_all_xgb_models()

    # Step 3: Evaluate All Models
    print(f"\n{'#' * 80}")
    print("# STEP 3: EVALUATE ALL MODELS")
    print(f"{'#' * 80}")
    testing_dir = os.path.join(BASE_DIR, "Testing")
    sys.path.insert(0, testing_dir)
    spec = importlib.util.spec_from_file_location(
        "evaluate_all_models", os.path.join(testing_dir, "evaluate_all_models.py")
    )
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)
    eval_mod.run_evaluation()

    # Step 4: Generate Visualizations
    print(f"\n{'#' * 80}")
    print("# STEP 4: GENERATE VISUALIZATIONS")
    print(f"{'#' * 80}")
    viz_dir = os.path.join(BASE_DIR, "Visualization")
    sys.path.insert(0, viz_dir)
    spec = importlib.util.spec_from_file_location(
        "generate_plots_v2", os.path.join(viz_dir, "generate_plots_v2.py")
    )
    viz_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz_mod)

    elapsed = time.time() - start
    print(f"\n{'=' * 80}")
    print(f"PIPELINE COMPLETE in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 80}")
    print(f"\nOutputs:")
    print(f"  Dataset:      d:\\Internproj\\new dataset\\engineered_ml_dataset.csv")
    print(f"  RF Models:    {os.path.join(BASE_DIR, 'Random Forest', '*.pkl')}")
    print(f"  XGB Models:   {os.path.join(BASE_DIR, 'XGBoost', '*.json')}")
    print(f"  Results:      {os.path.join(BASE_DIR, 'Testing', 'results')}")
    print(f"  Visualizations: d:\\Internproj\\Visualization\\v2_walkforward\\")


if __name__ == "__main__":
    run_pipeline()
