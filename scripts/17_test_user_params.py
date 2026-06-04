import numpy as np
import pandas as pd
from src.data_loader import load_feature_matrix
from src.ml_models import (
    add_target,
    apply_target_transform,
    evaluate_predictions,
    invert_target_transform,
    prepare_xy,
    time_train_test_split,
    train_xgboost,
)

params = {
    'n_estimators': 400,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'gamma': 0.5,
    'reg_alpha': 1.0,
    'reg_lambda': 3.0,
    'random_state': 42,
    'n_jobs': -1
}

df = load_feature_matrix()
df = add_target(df, scope='all')
train, test = time_train_test_split(df)

X_train, y_train, _ = prepare_xy(train, scope='all')
X_test, y_test, mask_test = prepare_xy(test, scope='all')

y_train_transformed = apply_target_transform(y_train, 'log1p')
model = train_xgboost(X_train, pd.Series(y_train_transformed), params=params)

y_pred_transformed = model.predict(X_test)
y_pred = np.clip(invert_target_transform(y_pred_transformed, 'log1p'), 0, None)

test_meta = test.loc[mask_test, ['country', 'week_start']].reset_index(drop=True)
pred_df = test_meta.copy()
pred_df['actual'] = y_test.to_numpy()
pred_df['predicted'] = y_pred

metrics = evaluate_predictions(pred_df)
print("=== User Custom Parameters Test Set Results ===")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE : {metrics['mae']:.4f}")
print(f"Directional Accuracy: {metrics['directional_accuracy']:.4f}")
print(f"Per-country dir acc: {metrics['per_country_directional']}")
