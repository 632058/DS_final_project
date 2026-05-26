import subprocess
import sys

branches = [
    ('--target-scope', 'all', '--target-transform', 'raw'),
    ('--target-scope', 'all', '--target-transform', 'log1p'),
    ('--target-scope', 'domestic', '--target-transform', 'raw'),
    ('--target-scope', 'domestic', '--target-transform', 'log1p'),
]

scripts = [
    'scripts/09_train_lasso.py',
    'scripts/10_train_xgboost.py',
    'scripts/10b_collate_model_comparison.py',
    'scripts/10c_per_country_xgboost_diagnostic.py',
    'scripts/11_compute_shap.py',
    'scripts/15_make_ml_diagnostic_figures.py',
]

for scope, scope_val, transform, transform_val in branches:
    print(f"\n==================================================")
    print(f"RUNNING PIPELINE FOR {scope_val}/{transform_val}")
    print(f"==================================================")
    for script in scripts:
        cmd = [sys.executable, script, scope, scope_val, transform, transform_val]
        print(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, env={'PYTHONPATH': '.'})
        if res.returncode != 0:
            print(f"Error executing {script} for {scope_val}/{transform_val}. Exiting.")
            sys.exit(1)
print("\nPipeline run completed successfully!")
