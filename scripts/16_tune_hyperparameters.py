"""Hyperparameter tuning script for Lasso and XGBoost models.

Uses a time-based validation split on the training set to prevent data leakage.
Finds the best hyperparameters, trains on the full training set, and reports test metrics.
Optimizes for both RMSE and Directional Accuracy.

Run:
    python scripts/16_tune_hyperparameters.py
"""
from __future__ import annotations

import argparse
import random
import numpy as np
import pandas as pd

from src.data_loader import load_feature_matrix
from src.ml_cli import add_branch_args, resolve_paths
from src.ml_models import (
    add_target,
    apply_target_transform,
    evaluate_predictions,
    invert_target_transform,
    prepare_xy,
    time_train_test_split,
    train_lasso,
    train_xgboost,
)

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


def main(scope: str, transform: str, n_iter: int = 150) -> None:
    print(f"=== Starting Multi-Objective Tuning for scope={scope}, transform={transform} ===")
    
    # 1. Load data and split train/test
    df = load_feature_matrix()
    df = add_target(df, scope=scope)
    train, test = time_train_test_split(df)

    # 2. Split train into sub_train and validation by time (80/20 split)
    train = train.sort_values('week_start').reset_index(drop=True)
    val_cutoff_idx = int(len(train) * 0.8)
    val_cutoff_week = train['week_start'].iloc[val_cutoff_idx]
    
    sub_train = train[train['week_start'] < val_cutoff_week].copy()
    val = train[train['week_start'] >= val_cutoff_week].copy()

    # Prepare features and targets
    X_sub_train, y_sub_train, _ = prepare_xy(sub_train, scope=scope)
    X_val, y_val, mask_val = prepare_xy(val, scope=scope)
    X_train, y_train, _ = prepare_xy(train, scope=scope)
    X_test, y_test, mask_test = prepare_xy(test, scope=scope)

    y_sub_train_transformed = apply_target_transform(y_sub_train, transform)
    y_train_transformed = apply_target_transform(y_train, transform)

    val_meta = val.loc[mask_val, ['country', 'week_start']].reset_index(drop=True)
    test_meta = test.loc[mask_test, ['country', 'week_start']].reset_index(drop=True)

    # ==========================================
    # Tuning Lasso
    # ==========================================
    print("\n--- Tuning Lasso baseline (Optimizing for Validation RMSE) ---")
    lasso_alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    best_lasso_alpha_rmse = None
    best_lasso_val_rmse = float('inf')
    best_lasso_rmse_metrics = {}

    best_lasso_alpha_dir = None
    best_lasso_val_dir = -1.0
    best_lasso_dir_metrics = {}

    for alpha in lasso_alphas:
        model, scaler = train_lasso(X_sub_train, pd.Series(y_sub_train_transformed), alpha=alpha)
        y_pred_val_transformed = model.predict(scaler.transform(X_val))
        y_pred_val = np.clip(invert_target_transform(y_pred_val_transformed, transform), 0, None)

        pred_df = val_meta.copy()
        pred_df['actual'] = y_val.to_numpy()
        pred_df['predicted'] = y_pred_val
        metrics = evaluate_predictions(pred_df)

        if metrics['rmse'] < best_lasso_val_rmse:
            best_lasso_val_rmse = metrics['rmse']
            best_lasso_alpha_rmse = alpha
            best_lasso_rmse_metrics = metrics

        if metrics['directional_accuracy'] > best_lasso_val_dir:
            best_lasso_val_dir = metrics['directional_accuracy']
            best_lasso_alpha_dir = alpha
            best_lasso_dir_metrics = metrics

    print(f"Lasso Best Alpha (by RMSE)   : {best_lasso_alpha_rmse} (Val RMSE={best_lasso_val_rmse:.4f}, dir_acc={best_lasso_rmse_metrics['directional_accuracy']:.4f})")
    print(f"Lasso Best Alpha (by Dir Acc): {best_lasso_alpha_dir} (Val RMSE={best_lasso_dir_metrics['rmse']:.4f}, dir_acc={best_lasso_val_dir:.4f})")

    # Evaluate both on Test set
    # 1. RMSE-optimized Lasso
    model_lasso_rmse, scaler_rmse = train_lasso(X_train, pd.Series(y_train_transformed), alpha=best_lasso_alpha_rmse)
    y_pred_rmse = np.clip(invert_target_transform(model_lasso_rmse.predict(scaler_rmse.transform(X_test)), transform), 0, None)
    pred_test_rmse = test_meta.copy()
    pred_test_rmse['actual'] = y_test.to_numpy()
    pred_test_rmse['predicted'] = y_pred_rmse
    test_metrics_lasso_rmse = evaluate_predictions(pred_test_rmse)

    # 2. Dir-optimized Lasso
    model_lasso_dir, scaler_dir = train_lasso(X_train, pd.Series(y_train_transformed), alpha=best_lasso_alpha_dir)
    y_pred_dir = np.clip(invert_target_transform(model_lasso_dir.predict(scaler_dir.transform(X_test)), transform), 0, None)
    pred_test_dir = test_meta.copy()
    pred_test_dir['actual'] = y_test.to_numpy()
    pred_test_dir['predicted'] = y_pred_dir
    test_metrics_lasso_dir = evaluate_predictions(pred_test_dir)

    print("\n--- Lasso Test Set Results ---")
    print(f"Lasso (Optimized for RMSE)   : RMSE={test_metrics_lasso_rmse['rmse']:.4f}, MAE={test_metrics_lasso_rmse['mae']:.4f}, dir_acc={test_metrics_lasso_rmse['directional_accuracy']:.4f}")
    print(f"Lasso (Optimized for Dir Acc): RMSE={test_metrics_lasso_dir['rmse']:.4f}, MAE={test_metrics_lasso_dir['mae']:.4f}, dir_acc={test_metrics_lasso_dir['directional_accuracy']:.4f}")

    # ==========================================
    # Tuning XGBoost
    # ==========================================
    print(f"\n--- Tuning XGBoost (Randomized Search, {n_iter} iterations) ---")
    
    # Define hyperparameter grid
    xgb_grid = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 800, 1000],
        'max_depth': [2, 3, 4, 5, 6, 7],
        'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
        'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 10, 15, 20, 30, 40, 50],
        'gamma': [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    }

    best_xgb_params_rmse = None
    best_xgb_val_rmse = float('inf')
    best_xgb_rmse_metrics = {}

    best_xgb_params_dir = None
    best_xgb_val_dir = -1.0
    best_xgb_dir_metrics = {}

    # Track joint (compromise) model: maximize Directional Accuracy while keeping RMSE reasonable (e.g. within 10% of best validation RMSE)
    best_xgb_params_joint = None
    best_xgb_val_joint_score = -1.0
    best_xgb_joint_metrics = {}

    for i in range(n_iter):
        params = {k: random.choice(v) for k, v in xgb_grid.items()}
        params['random_state'] = 42
        params['n_jobs'] = -1

        model = train_xgboost(X_sub_train, pd.Series(y_sub_train_transformed), params=params)
        y_pred_val_transformed = model.predict(X_val)
        y_pred_val = np.clip(invert_target_transform(y_pred_val_transformed, transform), 0, None)

        pred_df = val_meta.copy()
        pred_df['actual'] = y_val.to_numpy()
        pred_df['predicted'] = y_pred_val
        metrics = evaluate_predictions(pred_df)

        # 1. Update best RMSE
        if metrics['rmse'] < best_xgb_val_rmse:
            best_xgb_val_rmse = metrics['rmse']
            best_xgb_params_rmse = params
            best_xgb_rmse_metrics = metrics

        # 2. Update best Directional Accuracy
        if metrics['directional_accuracy'] > best_xgb_val_dir:
            best_xgb_val_dir = metrics['directional_accuracy']
            best_xgb_params_dir = params
            best_xgb_dir_metrics = metrics

        # 3. Update best Joint Score (e.g. directional accuracy - 0.01 * rmse)
        # This penalizes models that get high direction by making wildly unstable predictions
        joint_score = metrics['directional_accuracy'] - 0.01 * metrics['rmse']
        if joint_score > best_xgb_val_joint_score:
            best_xgb_val_joint_score = joint_score
            best_xgb_params_joint = params
            best_xgb_joint_metrics = metrics
            
        if (i + 1) % 30 == 0:
            print(f"  Iteration {i+1}/{n_iter}... best val RMSE: {best_xgb_val_rmse:.4f} | best val Dir Acc: {best_xgb_val_dir:.4f}")

    print("\n=== XGBoost Tuning Results on Validation Set ===")
    print(f"1. Best by RMSE       : Val RMSE={best_xgb_val_rmse:.4f}, Dir={best_xgb_rmse_metrics['directional_accuracy']:.4f}")
    print(f"2. Best by Dir Acc    : Val RMSE={best_xgb_dir_metrics['rmse']:.4f}, Dir={best_xgb_val_dir:.4f}")
    print(f"3. Best by Joint Score: Val RMSE={best_xgb_joint_metrics['rmse']:.4f}, Dir={best_xgb_joint_metrics['directional_accuracy']:.4f}")

    # Train and evaluate all three on Test set
    def evaluate_on_test(params, name):
        model = train_xgboost(X_train, pd.Series(y_train_transformed), params=params)
        y_pred = np.clip(invert_target_transform(model.predict(X_test), transform), 0, None)
        pred_test_df = test_meta.copy()
        pred_test_df['actual'] = y_test.to_numpy()
        pred_test_df['predicted'] = y_pred
        metrics = evaluate_predictions(pred_test_df)
        print(f"\n[{name} on Test Set]")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE : {metrics['mae']:.4f}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.4f}")
        print(f"  Per-country dir acc: {metrics['per_country_directional']}")
        return metrics, params

    res_rmse, p_rmse = evaluate_on_test(best_xgb_params_rmse, "XGBoost Optimized for RMSE")
    res_dir, p_dir = evaluate_on_test(best_xgb_params_dir, "XGBoost Optimized for Dir Acc")
    res_joint, p_joint = evaluate_on_test(best_xgb_params_joint, "XGBoost Optimized for Joint Score")

    # Load baseline for final comparison
    paths = resolve_paths(scope=scope, transform=transform)
    try:
        baseline_xgb_metrics = pd.read_parquet(paths.out_dir / "xgboost_metrics.parquet").iloc[0]
        print("\n=== Final Test Comparison ===")
        print(f"Baseline (Untuned)   : RMSE={baseline_xgb_metrics['rmse']:.4f}, MAE={baseline_xgb_metrics['mae']:.4f}, Dir={baseline_xgb_metrics['directional_accuracy']:.4f}")
        print(f"Tuned (RMSE-Opt)     : RMSE={res_rmse['rmse']:.4f}, MAE={res_rmse['mae']:.4f}, Dir={res_rmse['directional_accuracy']:.4f}")
        print(f"Tuned (Dir-Opt)      : RMSE={res_dir['rmse']:.4f}, MAE={res_dir['mae']:.4f}, Dir={res_dir['directional_accuracy']:.4f}")
        print(f"Tuned (Joint-Opt)    : RMSE={res_joint['rmse']:.4f}, MAE={res_joint['mae']:.4f}, Dir={res_joint['directional_accuracy']:.4f}")
        
        # Print best parameters of the models for their reference
        print("\n=== Best RMSE-Optimized Hyperparameters ===")
        for k, v in p_rmse.items():
            print(f"  {k}: {v}")
            
        print("\n=== Best Dir-Optimized Hyperparameters ===")
        for k, v in p_dir.items():
            print(f"  {k}: {v}")
            
        print("\n=== Best Joint-Optimized Hyperparameters (Recommended) ===")
        for k, v in p_joint.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"\nCould not load comparison metrics: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tune hyperparameters with multi-objective optimization.")
    parser.add_argument('--iter', type=int, default=150, help="Number of random search iterations.")
    add_branch_args(parser)
    args = parser.parse_args()
    main(scope=args.target_scope, transform=args.target_transform, n_iter=args.iter)
