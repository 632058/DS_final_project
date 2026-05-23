# Feature Scope 雙分支重構與 Fig 5/6 改善計畫

> 狀態：給 Claude Code 的下一輪實作規格
> 日期：2026-05-23
> 依據：已同步 `docs/issues_E.md` 最新進度
> 目標：承接目前 `protest_count_all` 主線，先補最有機會改善 Fig 5/6 的 protest 自迴歸特徵，再重構成 domestic / all 兩個可重現分支。

## 0. 目前進度，不要重複做

根據 `docs/issues_E.md` 最新狀態，目前已完成或已重新定性的事項如下。

### 已完成

1. **Issue #1 已解決**

   `model_comparison.parquet` 已改由 `scripts/10b_collate_model_comparison.py` 從 lasso / xgb metrics 自動拼出。之後流程維持：

   ```bash
   python scripts/09_train_lasso.py
   python scripts/10_train_xgboost.py
   python scripts/10b_collate_model_comparison.py
   ```

2. **Data path bug 已修**

   `src.data_loader.load_country_weekly()` 目前已讀：

   ```text
   output/country_weekly.parquet
   ```

   不要改回 `data/country_weekly.parquet`。

3. **目前 ML 主線已明確使用 `protest_count_all` target**

   `src/ml_models.add_target()` 現在使用：

   ```text
   protest_count_next_week = protest_count_all.shift(-1)
   ```

   這使得目前 ML 指標為：

   | Model | RMSE | MAE | Directional Accuracy |
   |---|---:|---:|---:|
   | Lasso | 12.48 | 6.88 | 64.2% |
   | XGBoost | 13.24 | 6.84 | 65.5% |

4. **原本的 scope mismatch 假設已被證偽**

   目前 `issues_E.md` 已重新定性：

   - LE 大幅改善主要來自 data path / aggregation 口徑修正。
   - TU 殘留 under-predict 不是因為 features OOD，而是缺少 protest 自迴歸特徵。

### 目前仍有的 reproducibility gap

`output/country_weekly.parquet` 目前有：

```text
protest_count
protest_count_all
```

但目前檢查到的 `scripts/01_build_country_weekly.py` 內容只產生 `protest_count`，沒有產生 `protest_count_all`。也就是說，**現在的 output schema 可能不是由目前 script 可重現產生的**。

下一輪第一件事應該先確認並修好這個 gap，避免重跑 `01_build_country_weekly.py` 後把 `protest_count_all` 洗掉。

## 1. 下次開工的優先順序

### Phase 1：最小修法，優先改善 Fig 5/6

這是短期最重要的工作。不要先大改整個 repo。

目標：在目前 `protest_count_all` 主線上，補上 protest 自迴歸特徵，重跑 ML pipeline，看 Fig 5/6 是否改善。

#### 1.1 先修 weekly build 的 schema 可重現性

確認 `scripts/01_build_country_weekly.py` 重跑後仍能產出：

```text
protest_count
protest_count_all
```

目前建議語義：

```text
protest_count      = domestic-only protest，短期保留舊名
protest_count_all  = all-scope protest，ML 目前主 target
```

若要更乾淨，可以同時新增：

```text
protest_count_domestic = protest_count
```

但短期不要移除 `protest_count`，避免 C/D/F 舊 scripts 直接壞掉。

#### 1.2 在 `feature_matrix` 補 `protest_count_all` 的 lag / rolling / diff

根據 `issues_E.md` 的 Task #3 診斷，TU 的最大問題是 feature matrix 90 個欄位中完全沒有：

```text
protest_count_all_lag*
protest_count_all_rolling_*
```

但 TU train 期間 `protest_count_all` lag-1 自相關約 0.53，是強訊號。

最小改法：

```python
ROLLING_LAG_TARGETS = [
    'avg_tone',
    'avg_goldstein',
    'n_material_conf',
    'protest_count_all',
]
```

這會自動產生：

```text
protest_count_all_lag1 ... lag12
protest_count_all_rolling_mean_4w / 8w / 12w
protest_count_all_rolling_std_4w / 8w / 12w
protest_count_all_diff_1w
protest_count_all_diff_4w
protest_count_all_pct_change_4w
```

#### 1.3 讓當週 `protest_count_all` 可作為 feature

目前 `src/ml_models.NON_FEATURE_COLS` 包含：

```python
'protest_count_all'
```

根據 `issues_E.md`，建議從 `NON_FEATURE_COLS` 移除它，讓當週 `protest_count_all` 進入 feature set。

這不是 leakage，因為 target 是：

```text
next_week = protest_count_all.shift(-1)
```

當週 protest count 用來預測下週 protest count，是合法的 autoregressive feature。

保留排除：

```python
'protest_count'
'protest_count_next_week'
```

#### 1.4 重跑最小 pipeline

短期先跑這條，不要先做完整雙分支：

```bash
python scripts/01_build_country_weekly.py
python scripts/04_build_feature_matrix.py
python scripts/sanity_check.py

python scripts/09_train_lasso.py
python scripts/10_train_xgboost.py
python scripts/10b_collate_model_comparison.py
python scripts/11_compute_shap.py
python scripts/15_make_ml_diagnostic_figures.py
```

重點檢查：

```text
output/feature_matrix.parquet 是否有 protest_count_all_lag1
XGBoost RMSE / MAE 是否改善
TU 2025-03-17 殘差是否縮小
LE 2026-03-02 over-predict 是否被低 protest_count_all 當週值對沖
Fig 5 是否更靠近對角線
Fig 6 residual outliers 是否縮小
```

### Phase 2：完整重構成 domestic / all 兩個分支

這是你想要的長期結構，應該做，但建議在 Phase 1 驗證 protest 自迴歸特徵有效後再做。

目標：同一份乾淨 `feature_matrix`，支援兩個 target branch：

```text
domestic branch -> protest_count_domestic_next_week
all branch      -> protest_count_all_next_week
```

## 2. 目標資料契約

### 2.1 Canonical Weekly Dataset

唯一 canonical weekly file：

```text
output/country_weekly.parquet
```

必要欄位：

```text
protest_count_domestic
protest_count_all
protest_count_crossborder
```

定義：

```text
protest_count_domestic    = domestic-only protest
protest_count_all         = all-scope protest
protest_count_crossborder = protest_count_all - protest_count_domestic
```

短期 alias：

```text
protest_count = protest_count_domestic
```

原因：C/D/F 目前還有很多 script 直接讀 `protest_count`。重構時可以保留 alias，但新程式應明確使用 domestic / all 欄位。

### 2.2 Conflict / Media 欄位

至少保留現有 all-scope 欄位：

```text
avg_tone
avg_goldstein
n_material_conf
```

中期建議新增更明確命名：

```text
avg_tone_all = avg_tone
avg_goldstein_all = avg_goldstein
n_material_conf_all = n_material_conf
```

如果時間足夠，再加 domestic / crossborder 版本：

```text
n_material_conf_domestic
n_material_conf_crossborder
avg_tone_domestic
avg_tone_crossborder
avg_goldstein_domestic
avg_goldstein_crossborder
```

這對 LE 的 OOD / 跨境戰爭問題會比較有解釋力。

## 3. 乾淨的 Feature Matrix 設計

只維護一份寬表：

```text
output/feature_matrix.parquet
```

不要建立兩份 feature matrix。用同一張寬表放 domestic / all / crossborder 欄位，再由模型 script 選 target。

### 必要 target-history features

All branch：

```text
protest_count_all
protest_count_all_lag1 ... lag12
protest_count_all_rolling_mean_4w / 8w / 12w
protest_count_all_rolling_std_4w / 8w / 12w
protest_count_all_diff_1w
protest_count_all_diff_4w
```

Domestic branch：

```text
protest_count_domestic
protest_count_domestic_lag1 ... lag12
protest_count_domestic_rolling_mean_4w / 8w / 12w
protest_count_domestic_rolling_std_4w / 8w / 12w
protest_count_domestic_diff_1w
protest_count_domestic_diff_4w
```

Crossborder controls：

```text
protest_count_crossborder
protest_count_crossborder_lag1 ... lag12
crossborder_protest_share
```

### Ratio features

保留現有：

```text
protest_ratio = protest_count_all / n_events
material_conf_ratio = n_material_conf / n_events
verbal_conf_ratio = n_verbal_conf / n_events
```

雙分支重構後建議改成：

```text
protest_ratio_all = protest_count_all / n_events
protest_ratio_domestic = protest_count_domestic / n_events
crossborder_protest_share = protest_count_crossborder / protest_count_all
```

## 4. 雙分支輸出設計

### 4.1 All Branch

這是目前 ML 主線，短期 report 也可先用它。

Target：

```text
protest_count_all_next_week
```

輸出：

```text
output/ml_results/all/raw/
output/ml_results/all/log1p/
figures/all/
```

### 4.2 Domestic Branch

這是之後要補的正式分支，對齊「國內抗議」研究問題。

Target：

```text
protest_count_domestic_next_week
```

輸出：

```text
output/ml_results/domestic/raw/
output/ml_results/domestic/log1p/
figures/domestic/
```

### 4.3 CLI 介面

建議新增：

```bash
python scripts/09_train_lasso.py --target-scope all --target-transform raw
python scripts/10_train_xgboost.py --target-scope all --target-transform raw
python scripts/10_train_xgboost.py --target-scope all --target-transform log1p

python scripts/09_train_lasso.py --target-scope domestic --target-transform raw
python scripts/10_train_xgboost.py --target-scope domestic --target-transform raw
python scripts/10_train_xgboost.py --target-scope domestic --target-transform log1p
```

mapping：

```text
all      -> protest_count_all
domestic -> protest_count_domestic
```

## 5. Fig 5/6 改善策略

目前 `issues_E.md` 的最新診斷顯示：

- TU 不是 OOD，而是缺 protest 自迴歸特徵。
- LE 殘留問題是真正 OOD 外推。

因此改善順序應該是：

### 5.1 第一優先：補 protest 自迴歸特徵

這是最可能立刻讓 Fig 5/6 變好的改動。

預期效果：

- TU 2025-03-17 這列：當週 `protest_count_all=99` 會變成強 feature，模型有機會把下週 197 預測到 100+。
- LE 2026-03-02 這列：當週 `protest_count_all` 很低，可以對沖 OOD conflict features 造成的 over-predict。

### 5.2 第二優先：`log1p` target transform

Protest count 是長尾 count data，可以新增：

```text
--target-transform raw|log1p
```

訓練：

```python
y_train_model = np.log1p(y_train)
```

預測：

```python
y_pred = np.expm1(model.predict(X_test))
y_pred = np.clip(y_pred, 0, None)
```

預期效果：

- 降低少數極端週對模型的支配。
- 通常能讓 Fig 5/6 的 outliers 收斂。
- 可能讓 RMSE 小幅改善，也可能主要改善視覺穩定性。

### 5.3 第三優先：LE OOD 處理

如果 LE 仍 over-predict，可加做：

1. **Train p99 clipping / winsorization**

   對這類特徵在 inference 前 clip：

   ```text
   n_material_conf
   n_verbal_conf
   n_material_conf_lag*
   n_material_conf_diff_*
   ```

   注意：clip threshold 必須只用 train set 算，避免 leakage。

2. **新增 domestic / crossborder conflict features**

   讓模型分辨：

   ```text
   n_material_conf_domestic
   n_material_conf_crossborder
   ```

   這比只靠 `n_material_conf` 更能處理 LE 這類跨境戰爭情境。

### 5.4 第四優先：per-country robustness

如果 pooled model 還是讓 LE / TU outliers 很大，可以加做：

```text
pooled_all_log1p
per_country_all_log1p
pooled_domestic_log1p
per_country_domestic_log1p
```

這主要作為 diagnostic，不一定當主結果。

## 6. Directional Accuracy 需要修

目前 `src/ml_models.evaluate()` 仍直接：

```python
np.diff(y_true.values)
```

在 pooled panel 中，這可能拿不同國家的相鄰 row 比方向。

建議修法：

- `evaluate()` 只算 RMSE / MAE。
- 新增 `evaluate_predictions(pred_df)`，用 `country + week_start + actual + predicted`，在每個 country 內部排序後算 directional accuracy。

這不是 Fig 5/6 視覺改善的核心，但會影響 report 指標可信度。

## 7. 建議重跑順序

### Phase 1：最小改善

```bash
python scripts/01_build_country_weekly.py
python scripts/04_build_feature_matrix.py
python scripts/sanity_check.py

python scripts/09_train_lasso.py
python scripts/10_train_xgboost.py
python scripts/10b_collate_model_comparison.py
python scripts/11_compute_shap.py
python scripts/15_make_ml_diagnostic_figures.py
```

### Phase 2：雙分支 + log1p

等 CLI 支援後：

```bash
python scripts/09_train_lasso.py --target-scope all --target-transform raw
python scripts/10_train_xgboost.py --target-scope all --target-transform raw
python scripts/10_train_xgboost.py --target-scope all --target-transform log1p

python scripts/09_train_lasso.py --target-scope domestic --target-transform raw
python scripts/10_train_xgboost.py --target-scope domestic --target-transform raw
python scripts/10_train_xgboost.py --target-scope domestic --target-transform log1p

python scripts/15_make_ml_diagnostic_figures.py --target-scope all --target-transform log1p
python scripts/15_make_ml_diagnostic_figures.py --target-scope domestic --target-transform log1p
```

## 8. Acceptance Criteria

### Phase 1

- `scripts/01_build_country_weekly.py` 可重生含 `protest_count_all` 的 `output/country_weekly.parquet`。
- `output/feature_matrix.parquet` 有 `protest_count_all_lag1` 等自迴歸特徵。
- `protest_count_all` 不再被 `prepare_xy()` 排除。
- XGBoost 重新訓練後，TU 2025-03-17 殘差應明顯縮小。
- Fig 5/6 重新產生，且 outliers 比目前版本少。

### Phase 2

- ML scripts 支援 `--target-scope all|domestic`。
- ML scripts 支援 `--target-transform raw|log1p`。
- `output/ml_results/all/` 與 `output/ml_results/domestic/` 分開保存。
- Fig 5/6 可分別產生 all / domestic 版本，不互相覆蓋。
- 每個 branch 都輸出 per-country metrics。
- Directional Accuracy 在 country 內部計算。

## 9. 與目前修法是否衝突

不衝突，但原本 plan 需要降級成分階段：

- 原本 plan 說「主結果改 domestic」，但目前 `issues_E.md` 已把短期 ML 主線定為 `protest_count_all`，且數字已改善。所以下次不應一開始就切 domestic，否則會打亂已修好的 Issue #2/#3 指標。
- 原本 plan 說「scope mismatch 是主因」，但最新診斷已證偽。文件現在改成：scope cleanup 是架構重構需求，不是目前 LE/TU 殘差的主要解釋。
- 原本 plan 說先做 domestic/all 分支；現在改成先補 `protest_count_all` 自迴歸特徵，再做雙分支。

結論：下一輪 Claude Code 應先做 Phase 1，確認 Fig 5/6 改善，再做 Phase 2 的雙分支重構。

## 10. Report 寫法建議

短期 all-target 主線：

> 修正資料路徑後，模型目標明確設定為 `protest_count_all`，使 target 與媒體 / 衝突結構特徵維持相同 all-event scope。XGBoost 與 Lasso 表現接近，LE 的 over-prediction 大幅收斂。

TU 殘留問題：

> 土耳其 Imamoglu 案例的主要殘差不是 OOD feature 問題，而是現行模型缺少 protest 自迴歸特徵。當週抗議數已從 7 跳到 99 時，模型仍無法將這個訊號傳遞到下週 197 的預測。

雙分支重構後：

> 後續分析將國內抗議與所有抗議分成 domestic / all 兩條分支。主文可依研究問題選擇 domestic 或 all，另一條作為 robustness check。
