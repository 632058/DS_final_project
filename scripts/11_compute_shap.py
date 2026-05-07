"""Compute SHAP values for the XGBoost model and save them.

Run: python scripts/11_compute_shap.py
Output:
- output/ml_results/shap_values.npy
- figures/fig8_shap_summary.png
"""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.constants import FIGURES_DIR, OUTPUT_DIR
from src.viz_template import apply_style


def main(max_display: int = 15) -> None:
    apply_style()

    artifact = joblib.load(OUTPUT_DIR / "ml_results" / "xgboost_model.joblib")
    model = artifact['model']
    X_test = pd.read_parquet(OUTPUT_DIR / "ml_results" / "X_test.parquet")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    np.save(OUTPUT_DIR / "ml_results" / "shap_values.npy", shap_values)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X_test, show=False, max_display=max_display)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    fig.savefig(FIGURES_DIR / "fig8_shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("SHAP summary saved to figures/fig8_shap_summary.png")


if __name__ == '__main__':
    main()
