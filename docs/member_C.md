# 成員 C 工作手冊

> 你負責**時序建模 ARIMAX**:對 5 國各跑一個 ARIMAX,以 `avg_tone`、`avg_goldstein`、`n_material_conf` 的 lag 作為外生變數,預測 `protest_count`。

---

## 全部交付物總覽

| 週次 | 交付物 | 對誰交付 | 截止 |
|---|---|---|---|
| W1 Day 1-3 | 熟悉 `src/arimax_model.py` + 跑通 `tests/test_arimax_model.py` | 自己 | Day 3 結束 |
| W1 Day 4-7 | ADF 平穩性檢定結果(5 國 × 3 序列) | 自己(決定差分階數用) | Day 7 結束 |
| W2 中 | ARIMAX 5 國建模(A 協助 2-3 國)+ Fig 7 | F | Week 2 中 |
| W2 末 | `report/methodology.md` 初稿 | F | Week 2 末 |
| W3 末 | 章節最終版 + 簡報 | 全組 | Week 3 末 |

---

## 依賴關係

- **你依賴 A**:Day 4 起需要 `output/country_weekly.parquet` 跑 ADF。
- **你依賴 B**:Week 2 需要 `output/feature_matrix.parquet`(用 lag 變數作 exog)。
- **A 協助你**:Week 2 你跟 A 分工跑 5 國 ARIMAX(建議 A 跑 CE/AR/CI、你跑 TU/LE)。
- **依賴你的人**:F(要 Fig 7)、D(可能參考你的 lag 選擇)。

---

## Week 1 Day 1-3:熟悉框架 + 跑通測試

### 你會用到的檔案
- `src/arimax_model.py`(已備好)
- `tests/test_arimax_model.py`(已備好)
- `scripts/05_run_adf.py`(Day 4 開始用)
- `scripts/06_run_arimax.py`(W2 開始用)

### 步驟

#### Step 1:讀懂 `src/arimax_model.py`
主要 function:
- `run_adf_test(series, name)` — ADF 平穩性檢定
- `select_difference_order(series)` — 找最小 d 使序列平穩
- `fit_arimax(y, exog)` — grid search (p, d, q) 用 AIC,d 由 ADF 決定
- `diagnose_residuals(residuals)` — Ljung-Box 殘差白噪檢定
- `run_country_arimax(country, df_features, output_dir)` — 一個國家的完整 pipeline:filter → exog → 80/20 切 → fit → forecast → save

#### Step 2:跑測試
- 執行 `pytest tests/test_arimax_model.py -v`
- 應全綠,確認:
  - random walk 不平穩(ADF 不顯著)
  - white noise 平穩(ADF 顯著)
  - random walk 的 d=1
  - 殘差白噪檢測

#### Step 3:複習 ARIMAX 概念
- `(p, d, q)`:p = AR 階、d = 差分階、q = MA 階
- exog = 外生變數,本專案用 `avg_tone`、`avg_goldstein`、`n_material_conf` 的 lag(避免 look-ahead)
- 預設 `exog_lag_set = (1, 2, 4)`,只用 lag 1, 2, 4(避免共線性)

#### Step 4:思考參數
- 若 D 的 CCF 結果顯示某國 best lag 是 5 而不是 4,你可以在 W2 改 `exog_lag_set=(1, 3, 5)` 跑該國

---

## Week 1 Day 4-7:ADF 平穩性檢定

### 步驟

#### Step 1:跑 ADF
- 執行 `python scripts/05_run_adf.py`
- 預期產出 `output/arimax_results/adf_test.parquet`
- 印出 5 國 × 3 序列(`protest_count`、`avg_tone`、`avg_goldstein`)的 p-value 與 `suggested_d`

#### Step 2:解讀結果
- `p_value < 0.05` → 平穩
- `suggested_d` 通常 0 或 1
- 若某序列 d=2 仍非平穩,序列可能有結構性問題(看 ACF/PACF 圖檢查)

#### Step 3(可選):畫 ACF / PACF
- 在 `notebooks/C_arimax/` 開 notebook
- 用 `statsmodels.graphics.tsaplots.plot_acf` 畫每國 `protest_count` 的 ACF, PACF
- 存 `figures/acf_pacf_diagnostic.png`(可選,若放報告 limitation 段)

---

## Week 2:ARIMAX 5 國 + Fig 7 + 章節

### Task 1:跟 A 約定分工(W2 開始前)

- **建議 A 跑**:CE、AR、CI
- **你跑**:TU、LE
- 確認後雙方執行 `scripts/06_run_arimax.py`

### Task 2:執行 ARIMAX
- A 跑:`python scripts/06_run_arimax.py CE AR CI`
- 你跑:`python scripts/06_run_arimax.py TU LE`
- 產出在 `output/arimax_results/<country>.json` + `output/arimax_results/<country>_forecast.parquet`
- 印出每國 (p,d,q)、AIC、RMSE、Ljung-Box p-value

### Task 3:檢查殘差診斷
- 看每國 `ljung_box_pvalue`,理想 > 0.05(殘差白噪)
- 若某國 < 0.05,殘差有自相關 → 模型不夠好
  - 增加 exog lag(改 `exog_lag_set=(1, 2, 3, 4, 6)`)
  - 增加 max_p / max_q(改 `fit_arimax` 呼叫)
  - 在 limitation 段討論

### Task 4:產出 Fig 7
- 等所有 5 國的 `_forecast.parquet` 都到位
- 執行 `python scripts/16_make_arimax_forecast_figure.py`
- 產出 `figures/fig7_arimax_forecast.png`(藍=實際、紅=預測、紅帶=95% CI)

### Task 5:撰寫 `report/methodology.md` 初稿(約 1500 字)

骨架已在 `report/methodology.md`,填內容:
1. **方法選擇理由**:ARIMAX 古典、可解釋、可加外生變數
2. **平穩性檢定**:5 國 × 3 序列 ADF 表
3. **模型階數選擇**:AIC grid search 流程
4. **外生變數**:lag 選擇理由(對齊 D 的 CCF 結果或標準 lag)
5. **模型診斷**:Ljung-Box 結果表
6. **預測表現**:5 國 RMSE / MAE 表 + 引用 Fig 7
7. **Limitation**:ARIMAX 假設線性、需平穩

---

## Week 3:章節最終版 + 簡報

### 跟其他成員對齊
- 跟 D 對比:CCF 找的 lag 是否跟你的 ARIMAX exog lag 一致?若不一致,在報告討論
- 跟 E 對比:ARIMAX 跟 XGBoost 哪一國表現差異最大?(可做一張比較表)

### 簡報重點
- 為什麼選 ARIMAX 而非 ARIMA(外生變數是核心訊號)
- 哪一國訊號最清楚(看 RMSE / Ljung-Box)
- 一張總表:5 國的 (p,d,q)、AIC、Ljung-Box p、RMSE

---

## 常見陷阱

| 陷阱 | 後果 | 解法 |
|---|---|---|
| ARIMAX exog 用當週值 | Look-ahead | 已用 lag 變數作 exog(`exog_lag_set=(1,2,4)`) |
| 隨機切 train/test | 時序不能 shuffle | `run_country_arimax` 已用 `iloc[:n_train]` 時間切 |
| 全用 lag 1-12 當 exog | 共線性嚴重,模型不收斂 | 預設 lag 1, 2, 4 |
| 5 國 pooling 跑 ARIMAX | ARIMAX 不易加 country fixed effect | 5 國各自獨立跑(已實作) |
| `enforce_stationarity=True` 不收斂 | SARIMAX 預設嚴格 | 已設 `False` 容錯 |
| 忘記做殘差診斷 | 報告會被質疑 | `run_country_arimax` 自動跑 Ljung-Box |
| 改 exog 但忘記改 `feature_matrix` | exog 欄位不存在 | 跟 B 確認 lag 欄位都已存在於 `feature_matrix` |

---

## 交付檢查清單

- [ ] **W1 Day 3**:`pytest tests/test_arimax_model.py` 全綠 + 讀完 `src/arimax_model.py`
- [ ] **W1 Day 7**:`output/arimax_results/adf_test.parquet`(5 國 × 3 序列)
- [ ] **W2 中**:跟 A 約好分工 + 自己負責的 2 國 ARIMAX 跑完
- [ ] **W2 末**:5 國 ARIMAX 結果 + Fig 7 + `report/methodology.md` 初稿
- [ ] **W3 末**:章節最終版 + 簡報投影片
