# 成員 D 工作手冊

> 你負責**統計檢定**:用 Cross-Correlation Function (CCF) 量化「媒體語氣領先抗議多少週」,用 Granger Causality Test 檢驗預測性因果。Week 2 後段協助 E 跑 SHAP。

---

## 全部交付物總覽

| 週次 | 交付物 | 對誰交付 | 截止 |
|---|---|---|---|
| W1 Day 1-3 | 熟悉 `src/stat_tests.py` + 跑通 `tests/test_stat_tests.py` | 自己 | Day 3 結束 |
| W1 Day 4-7 | 初步 CCF 結果 + 通知 C 各國 best lag | C | Day 7 結束 |
| W2 中 | Granger Causality 5 國 + Fig 4 | F | Week 2 中 |
| W2 末 | 協助 E 跑 SHAP(產出 Fig 8) | E、F | Week 2 末 |
| W2 末 | `report/statistics.md` 初稿 | F | Week 2 末 |
| W3 末 | 章節最終版 + 簡報 | 全組 | Week 3 末 |

---

## 依賴關係

- **你依賴 A**:Day 4 起需要 `output/country_weekly.parquet`。
- **你協助 E**:Week 2 末 E 跑完 XGBoost 後,你跑 SHAP(計算量大,所以拆出來給你)。
- **依賴你的人**:F(要 Fig 4)、C(可參考你的 lag 結果)、E(SHAP 結果合作)。

---

## Week 1 Day 1-3:熟悉框架 + 跑通測試

### 你會用到的檔案
- `src/stat_tests.py`(已備好)
- `tests/test_stat_tests.py`(已備好)
- `scripts/07_compute_ccf.py`(Day 4 開始用)
- `scripts/08_run_granger.py`(W2 開始用)
- `scripts/11_compute_shap.py`(W2 末用)

### 步驟

#### Step 1:讀懂 `src/stat_tests.py`
主要 function:
- `compute_ccf(x, y, max_lag)` — 計算 corr(x_{t-k}, y_t) for k = 0..max_lag,正向 lag = x 領先 y
- `granger_test(y, x, max_lag)` — H₀: x does NOT Granger-cause y,回報 best lag 的 p-value
- `stationarity_diff(series)` — 差分到平穩,回傳 (差分後序列, d)

#### Step 2:跑測試
- 執行 `pytest tests/test_stat_tests.py -v`
- 應全綠,確認:
  - `y = x.shift(3)` 時 CCF 在 lag=3 達峰值
  - 獨立序列不拒絕 H₀
  - 依賴序列(y 由 lagged x 構成)拒絕 H₀

#### Step 3:複習概念
- **CCF**:純相關,無因果意涵,但能找出最佳 lead time
- **Granger**:**predictive causality**(不是真因果) — x 的過去值對預測 y 有沒有額外貢獻
- **平穩性前置**:Granger 假設序列平穩,所以前面要差分

---

## Week 1 Day 4-7:初步 CCF

### 步驟

#### Step 1:跑 CCF
- 執行 `python scripts/07_compute_ccf.py`
- 預期產出三個檔(每個 X 變數一個):
  - `output/ccf_granger/ccf_avg_tone_to_protest.parquet`
  - `output/ccf_granger/ccf_avg_goldstein_to_protest.parquet`
  - `output/ccf_granger/ccf_n_material_conf_to_protest.parquet`
- 印出每國 best lag

#### Step 2:解讀方向
- 理論上 negative tone(語氣負面) → protest 增加,所以 `avg_tone → protest` 應該**負相關**
- 若是正相關,在報告解釋

#### Step 3:通知 C
- 群組通知:
  > @C CCF 跑完了。avg_tone → protest 的 best lag 是:CE=X、AR=Y、CI=Z、TU=W、LE=V。建議 ARIMAX 的 `exog_lag_set` 用這些 lag 而非預設的 (1, 2, 4)。

---

## Week 2:Granger + Fig 4 + 協助 E + 章節

### Task 1:跑 Granger Causality
- 執行 `python scripts/08_run_granger.py`
- 預期產出 `output/ccf_granger/granger_summary.parquet`
- 5 國 × 3 變數 = 15 個檢定結果,內含 `d_y`、`d_x`、`best_lag`、`best_p_value`、`reject_h0`

### Task 2:產出 Fig 4(CCF heatmap)
- 執行 `python scripts/14_make_ccf_heatmap.py`
- 預期產出 `figures/fig4_ccf_heatmap.png`
- 用 `avg_tone` 的 CCF(主要訊號);若想加 `avg_goldstein` 版本,複製 script 改 input 即可

### Task 3:協助 E 跑 SHAP

E 在 W2 中後段跑完 XGBoost,把模型存到 `output/ml_results/xgboost_model.joblib`,你接著跑:

- 執行 `python scripts/11_compute_shap.py`
- 預期產出:
  - `output/ml_results/shap_values.npy`(SHAP 數值給 E 之後分析)
  - `figures/fig8_shap_summary.png`(SHAP summary plot)
- 跟 E 確認:你產出後通知 E 在 ML 章節引用 Fig 8

> SHAP 計算約 5-30 分鐘(視 X_test 大小),可以一邊跑一邊寫章節。

### Task 4:撰寫 `report/statistics.md` 初稿(約 1500 字)

骨架已在 `report/statistics.md`,填內容:
1. **CCF 介紹**:定義、解讀方式
2. **Granger 介紹**:H₀、F-test、平穩性前置
3. **CCF 結果**:5 國 best lag 表 + 引用 Fig 4
4. **Granger 結果**:5 國 × 3 變數 p-value 表,標 `***` < 0.001、`**` < 0.01、`*` < 0.05
5. **與 ARIMAX 對比**:CCF 找的 lag 是否跟 C 用的一致
6. **討論 / Limitation**:多重檢定校正、樣本量、未做 multi-country panel Granger

---

## Week 3:章節最終版 + 簡報

### 簡報重點
- 一張總表:5 國 × 3 變數 × Granger p-value
- Fig 4 CCF heatmap
- 「最佳 lead time」的政策意涵:例如 LE 的最佳 lag = 4,代表新聞語氣可以提前 4 週警示

### Limitation 段落
- Granger Causality 是 predictive causality,不是真因果
- 樣本約 430 週,對 lag > 8 的 Granger 檢力不足
- 5 國各自跑,未做 multi-country panel Granger

---

## 常見陷阱

| 陷阱 | 後果 | 解法 |
|---|---|---|
| Granger 沒做平穩性前置 | 結果可能虛假顯著 | `08_run_granger.py` 已用 `stationarity_diff` 自動差分 |
| CCF 沒 standardize | 數值被 scale 主導 | `compute_ccf` 已 z-score |
| 沒對齊兩序列 index | shift 後對齊錯誤 | `compute_ccf` 已用 `intersection` |
| Granger 跨國 pooling | 假設違反(每國動態不同) | 每國獨立跑(已實作) |
| 多重檢定沒校正 | 5 國 × 12 lag 易 false positive | 在 limitation 段討論 Bonferroni / FDR |
| `grangercausalitytests` 預設 verbose=True | 印一堆東西 | 已設 `verbose=False` |
| CCF 結果跟 ARIMAX exog lag 不一致 | 報告自相矛盾 | 跟 C 對齊,要嘛 ARIMAX 用 CCF best lag,要嘛分開解釋 |
| SHAP 對全資料跑 | 30+ 分鐘 | 預設對 X_test(已縮減);若仍太慢,改用 X_test sample 1000 列 |

---

## 交付檢查清單

- [ ] **W1 Day 3**:`pytest tests/test_stat_tests.py` 全綠 + 讀完 `src/stat_tests.py`
- [ ] **W1 Day 7**:`output/ccf_granger/ccf_*.parquet`(3 個檔)+ 通知 C 各國 best lag
- [ ] **W2 中**:`output/ccf_granger/granger_summary.parquet` + Fig 4
- [ ] **W2 末**:`figures/fig8_shap_summary.png` + `output/ml_results/shap_values.npy` + `report/statistics.md` 初稿
- [ ] **W3 末**:章節最終版 + 簡報投影片
