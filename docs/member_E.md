# 成員 E 工作手冊

> 你負責**機器學習**:5 國 pooling 預測下週 `protest_count`(迴歸任務),Lasso baseline + XGBoost 主模型,SHAP 由 D 協助。

---

## 全部交付物總覽

| 週次 | 交付物 | 對誰交付 | 截止 |
|---|---|---|---|
| W1 Day 1-3 | 熟悉 `src/ml_models.py` + 跑通 `tests/test_ml_models.py` | 自己 | Day 3 結束 |
| W1 Day 4-7 | Lasso baseline + 對 B 提補強建議 | B | Day 7 結束 |
| W2 中 | XGBoost 主模型 + Fig 5 + Fig 6 | F | Week 2 中 |
| W2 末 | 通知 D 跑 SHAP(產出 Fig 8)+ `report/ml.md` 初稿 | D、F | Week 2 末 |
| W3 末 | 章節最終版 + per-country metrics + 簡報 | 全組 | Week 3 末 |

---

## 依賴關係

- **你依賴 B**:Day 4 起需要 `output/feature_matrix.parquet` v1,Week 2 中需要 v2。
- **D 協助你**:Week 2 末 D 跑 SHAP,你產出模型 `.joblib` 給 D。
- **依賴你的人**:F(要 Fig 5, 6, 8)。

---

## Week 1 Day 1-3:熟悉框架 + 跑通測試

### 你會用到的檔案
- `src/ml_models.py`(已備好)
- `tests/test_ml_models.py`(已備好)
- `scripts/09_train_lasso.py`(Day 4 開始用)
- `scripts/10_train_xgboost.py`(W2 開始用)

### 步驟

#### Step 1:讀懂 `src/ml_models.py`
主要 function:
- `add_target(df)` — 新增 `protest_count_next_week` = `protest_count.shift(-1)` per country
- `time_train_test_split(df, train_ratio=0.8)` — **時間切**(不是 shuffle),所有國共用同一 cutoff
- `prepare_xy(df)` — drop 非特徵欄(week_start, country, protest_count, target)+ drop NaN 列
- `evaluate(y_true, y_pred)` — RMSE、MAE、Directional Accuracy
- `train_lasso(X, y, alpha)` — 含 StandardScaler
- `train_xgboost(X, y, params)` — 不需要 scale

#### Step 2:跑測試
- 執行 `pytest tests/test_ml_models.py -v`
- 應全綠,確認:
  - `add_target` per country shift -1
  - `time_train_test_split` train.max < test.min(無 leak)
  - `prepare_xy` 移除 protest_count 等非特徵欄
  - `evaluate` 完美預測 RMSE = 0、Directional Accuracy = 1.0

#### Step 3:複習關鍵設計
- **任務性質**:迴歸,預測連續值(不是分類,不要用 ROC/AUC)
- **5 國 pooling**:全部混在一起訓練,加 country one-hot 給模型參考
- **時間切**:全資料按 `week_start` 排序後切前 80% / 後 20%
  - **絕對不能 `shuffle=True`**
  - **不能 per-country 切**(同一週可能 train + test 都有,leak)

---

## Week 1 Day 4-7:Lasso baseline

### 步驟

#### Step 1:確認 B 的 v1 已到位
- 看 `output/feature_matrix.parquet` 是否存在
- 跑 sanity check 確認

#### Step 2:跑 Lasso baseline
- 執行 `python scripts/09_train_lasso.py`
- 預期產出:
  - `output/ml_results/lasso_baseline.joblib`(model + scaler + feature_cols)
  - `output/ml_results/lasso_metrics.parquet`
- 印出 RMSE、MAE、Directional Accuracy

#### Step 3:檢視結果
- 開 notebook 載入 `lasso_baseline.joblib`
- 看哪些 feature 被選中(`model.coef_ != 0`)
- 排序看 top 10 重要 feature

#### Step 4:給 B 補強建議
- 群組通知範例:
  > @B Lasso baseline 跑完,RMSE=X、MAE=Y、Directional Acc=Z%。被選中的 top feature 集中在 lag1-4。建議 v2 加:tone × goldstein 互動項、n_material_conf 比例(/n_events)、季節性(week_of_year)。

#### Step 5(可選):alpha grid search
- 如有時間,可在 notebook 用 `TimeSeriesSplit(n_splits=5)` 做 CV
- 試 `alpha = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]`
- 選最佳 alpha 後重跑 baseline

---

## Week 2:XGBoost 主模型 + Fig 5/6 + 章節

### Task 1:跑 XGBoost
- 等 B 交 v2(若有補強)後執行 `python scripts/10_train_xgboost.py`
- 預期產出:
  - `output/ml_results/xgboost_model.joblib`(模型 + feature_cols,給 D 跑 SHAP 用)
  - `output/ml_results/X_test.parquet`、`y_test.parquet`(D 用)
  - `output/ml_results/xgboost_predictions.parquet`(含 country、week_start 給 F 用)
  - `output/ml_results/xgboost_metrics.parquet`

### Task 2(可選):Hyperparameter tuning
- 預設參數可能已不錯。若想調:在 notebook 用 `TimeSeriesSplit` 做 CV
- 簡化 grid:
  - `max_depth`: [3, 5, 7]
  - `n_estimators`: [200, 500]
  - `learning_rate`: [0.03, 0.05, 0.1]
- 選最佳後改 `scripts/10_train_xgboost.py` 的 `DEFAULT_PARAMS` 重跑

### Task 3:跨模型對比
- 開 notebook 比 Lasso baseline vs XGBoost 的 RMSE / MAE / Directional Acc
- 存 `output/ml_results/model_comparison.parquet`

### Task 4:產出 Fig 5, Fig 6
- 執行 `python scripts/15_make_ml_diagnostic_figures.py`
- 預期產出:
  - `figures/fig5_pred_vs_actual.png`(Predicted vs Actual 散點 + 對角線)
  - `figures/fig6_residuals.png`(Residual vs Predicted + Residual 分布)

### Task 5:通知 D 跑 SHAP
- 群組通知:
  > @D XGBoost 模型已存 `output/ml_results/xgboost_model.joblib`,X_test 在 `output/ml_results/X_test.parquet`。可以跑 `python scripts/11_compute_shap.py` 產出 Fig 8。

### Task 6:撰寫 `report/ml.md` 初稿(約 2000 字)

骨架已在 `report/ml.md`,填內容:
1. **任務定義**:迴歸 / 預測下週 protest_count / 5 國 pooling
2. **資料切分**:時間切 80/20、為何不能隨機切
3. **Baseline (Lasso)**:alpha 選擇、CV 結果、選中的 feature
4. **主模型 (XGBoost)**:超參數、test metrics
5. **Fig 5, 6**:預測表現視覺化
6. **可解釋性 (SHAP)**:引用 Fig 8(等 D 產出)+ top features 解讀
7. **與時序模型對比**:跟 C 的 ARIMAX 對齊
8. **Limitation**:樣本量、5 國差異、未做 multi-step forecast

---

## Week 3:章節最終版 + 簡報

### Per-country metrics(建議補)
在 notebook 中:
- 從 `xgboost_predictions.parquet` 含 `country` 欄
- 對每國分別算 RMSE / MAE
- 加進報告 Results 章節:某國訊號最強?某國最弱?

### 簡報重點
- 模型對比表(Lasso vs XGBoost vs ARIMAX,跟 C 對齊)
- Fig 5(最直覺的「我們預測得怎樣」)
- Fig 8 SHAP(哪些 feature 最重要)
- 哪一國表現最好 / 最差(per-country metrics)

---

## 常見陷阱

| 陷阱 | 後果 | 解法 |
|---|---|---|
| `train_test_split(shuffle=True)` | Look-ahead | `time_train_test_split` 已內建,不要自己用 sklearn |
| Per-country 切 80/20 | 同一週 train+test 都有 → leak | 已用 global week_start 切 |
| 忘記 `shift(-1)` | 預測「當週」而非「下週」 | `add_target` 已內建 per country shift -1 |
| Lasso 沒 scale | L1 偏向大 scale 特徵 | `train_lasso` 已內建 StandardScaler |
| XGBoost CV 用 KFold | 隨機切違反時序 | 用 `sklearn.model_selection.TimeSeriesSplit` |
| Y 含 0 用 MAPE | 除以 0 inf | `evaluate` 用 RMSE / MAE 不用 MAPE |
| 沒做 baseline 直接 XGBoost | 改善幅度說不出來 | 一定要有 Lasso baseline |
| SHAP 對全資料跑 | 30+ 分鐘 | D 已用 X_test(已縮減) |
| 把 `country` 當 feature 給 XGBoost | 字串無法訓練 | `prepare_xy` 已 drop `country`,用 `is_<country>` one-hot |

---

## 交付檢查清單

- [ ] **W1 Day 3**:`pytest tests/test_ml_models.py` 全綠 + 讀完 `src/ml_models.py`
- [ ] **W1 Day 7**:`output/ml_results/lasso_baseline.joblib` + metrics + 通知 B 補強建議
- [ ] **W2 中**:`output/ml_results/xgboost_model.joblib` + Fig 5 + Fig 6
- [ ] **W2 末**:通知 D 跑 SHAP(取得 Fig 8)+ `report/ml.md` 初稿
- [ ] **W3 末**:章節最終版 + per-country metrics + 簡報投影片
