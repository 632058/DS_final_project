# 分析方法審查筆記

> 撰寫：成員 B｜目的：整理各成員方法的潛在問題與建議，供 W2 對齊討論使用

---

## D：CCF + Granger Causality

### 問題 1：CCF 沒有差分前處理（重要）

`compute_ccf` 只做 z-score，沒有先做 ADF 再差分。若 `avg_tone` 與 `protest_count` 在 64 週窗口內仍有趨勢（例如語氣持續惡化、抗議持續攀升），計算出的 cross-correlation 可能是虛假相關（Yule's spurious correlation）。

Granger 那邊有用 `stationarity_diff` 先差分（對的），CCF 卻沒有，兩個方法邏輯不一致。

**建議**：CCF 前先對兩個序列各做 ADF，若不平穩則差分後再算。或在報告中明確說明為何不差分（例如：窗口已足夠短、事件研究設計假設窗口內序列局部平穩）。

---

### 問題 2：best_lag 選取 = 多重檢定（重要）

Granger 對每個變數做 1~12 lag 的 F-test，然後回報 p-value 最小的那個 lag。這等同於做了 12 次檢定後挑最顯著的，會嚴重膨脹 Type I error。

若 α = 0.05，Bonferroni 校正後應採 α' = 0.05 / 12 ≈ **0.004**。

對照 `summary_D.md` 的結果：

| 國家 | 變數 | 原 p-value | 通過 Bonferroni？ |
|---|---|---|---|
| 斯里蘭卡 | avg_tone | 0.004 | 剛好通過 |
| 智利 | avg_tone | 0.023 | ❌ 不通過 |
| 智利 | avg_goldstein | 0.016 | ❌ 不通過 |
| 黎巴嫩 | avg_tone | 0.003 | ✅ 通過 |
| 黎巴嫩 | avg_goldstein | 0.003 | ✅ 通過 |
| 黎巴嫩 | n_material_conf | 2e-7 | ✅ 通過 |

智利的兩個顯著結果在校正後消失，需在報告中說明。

**建議**：至少在 Limitation 段明確列出 Bonferroni 校正的結果，說明哪些結論受多重檢定影響。

---

### 問題 3：事件研究設計沒有評估 false positive rate（中等）

CCF/Granger 只在已知爆發日的 64 週窗口內跑，等於是「事後確認訊號存在」，但沒有回答：

> 在**沒有**爆發的時期，avg_tone 下滑也會出現類似的 CCF 模式嗎？

如果在和平時期也常見負相關，那 lag_8 的訊號就沒有預測價值。

**建議**（可做可不做，視時間而定）：從完整時序中隨機抽取同樣長度的非爆發期窗口，計算 CCF 分布，和爆發期的結果做比較。

---

### 問題 4：Granger 樣本量偏小（輕微）

64 週窗口，max_lag = 12，差分後有效觀測值約 50 筆，對 lag > 8 的 Granger 統計檢力不足（type II error 偏高）。

`summary_D.md` 已在限制段提到，繼續保留即可。

---

### 問題 5：n_material_conf 內生性（D 已知，確認處理方式）

D 已標注「n_material_conf 與 Y 高度重疊，顯著結果不代表獨立前兆」。建議主結論聚焦 avg_tone 和 avg_goldstein，n_material_conf 的結果移至 robustness check。

---

## C：ARIMAX

### 問題 1：exog_lag_set 預設 (1, 2, 4) 應參考 D 的 CCF 結果

目前預設用 lag 1, 2, 4，但 D 的 CCF 顯示黎巴嫩的最佳 lag 是 8–9 週。如果 C 堅持用 lag 1, 2, 4 給黎巴嫩，模型拿不到最強的外生訊號。

**建議**：D 在 W1 末通知各國 best lag 後，C 依照那個 lag 調整各國的 `exog_lag_set`。

---

### 問題 2：ARIMAX 預測值可能出現負數

`protest_count` 是計數資料（非負），但 ARIMAX 不限制預測值的符號，有機率輸出負值。

**建議**：在報告中說明此限制；若有負值預測，clip 到 0 後再算 RMSE/MAE。

---

### 問題 3：結構性斷點（COVID）

時序橫跨 2015–2026，COVID 期間（2020–2021）各國新聞量和社會動盪模式都有突變。ADF 可能因此誤判平穩性，影響 d 的選擇。

**建議**：在 `output/arimax_results/adf_test.parquet` 出來後，若某國 d=2 仍不平穩，檢查是否跟 COVID 窗口有關。

---

## E：ML（Lasso + XGBoost）

### 問題 1：用 MSE 預測計數資料（中等）

`protest_count` 是非負整數，且許多週為 0（零膨脹分佈）。MSE 損失在這類資料上會低估高值週期、高估零值週期。

**建議**：
- 試試 `log(1 + protest_count)` 作為訓練 target（最簡單）
- 或 XGBoost 改 `objective='count:poisson'`
- 評估時兩個版本都報 RMSE，讓讀者比較

---

### 問題 2：5 國 pooling 假設各國動態相似

土耳其（政治任命型抗議）和黎巴嫩（社會積累型抗議）的動態根本不同，D 的 CCF 結果也顯示這一點。把它們丟進同一個模型可能稀釋訊號。

**建議**：W3 補充 per-country 的 RMSE，如果某國表現特別差（例如土耳其），在報告中和 D 的正相關結果呼應解釋。

---

## 跨成員問題

### protest_count 定義不一致

| 成員 | protest_count 定義 |
|---|---|
| D | `country_1 == country_2`（國內抗議） |
| B → C、E | `country_1` 為目標國的所有抗議（含跨國） |

D 的 `summary_D.md` 指出「跨國抗議占土耳其所有抗議的 74%」，這個差異相當大，可能導致 ARIMAX 和 ML 在預測的其實是「包含跨國的protest_count」，而 D 分析的是「純國內」。

**建議**：B 在 `feature_matrix` 中加一個 `protest_count_domestic` 欄位（`country_1 == country_2` 的版本），讓 C 和 E 選擇要用哪個版本，並在報告中統一。

---

## 總結

| 成員 | 問題 | 嚴重程度 | 建議處理方式 |
|---|---|---|---|
| D | CCF 沒有差分 | 🔴 高 | CCF 前做 ADF，或在報告說明假設 |
| D | Granger best_lag 多重檢定 | 🔴 高 | 報告加 Bonferroni 校正結果 |
| D | 無 false positive rate 基準 | 🟡 中 | 可補隨機窗口對照，或列為 limitation |
| C | exog_lag_set 未依 CCF 調整 | 🟡 中 | D 通知 best lag 後 C 更新 |
| C | 預測值可能為負 | 🟢 低 | clip + 在報告說明 |
| E | MSE on count data | 🟡 中 | 試 log(1+y) 版本並比較 |
| E | 5 國 pooling 動態差異 | 🟡 中 | W3 補 per-country metrics |
| B+C+E | protest_count 定義不一致 | 🔴 高 | B 加 domestic 欄位，全組統一 |
