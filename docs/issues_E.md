# Issue Board — Member E (ML)

> Owner: Member E（曾弘舟）
> Status snapshot: 2026-05-23
> Scope: `scripts/09_train_lasso.py`, `scripts/10_train_xgboost.py`, `scripts/11_compute_shap.py`, `output/ml_results/*`

下列三項是目前 ML 結果在進到 `report/ml.md` 之前必須處理的問題。

---

## Issue #1 — `model_comparison.parquet` 與 `xgboost_metrics.parquet` 數字對不上

**Severity**: 嚴重 / 必修
**Status**: Open

### 敘述

`output/ml_results/` 底下兩個檔案對 XGBoost 報出來的指標完全不一致：

| 來源檔案 | RMSE | MAE | Directional Accuracy |
|---|---|---|---|
| `xgboost_metrics.parquet`（由 `scripts/10_train_xgboost.py` 自動產出） | **23.05** | 9.19 | 64.8% |
| `model_comparison.parquet`（沒有任何 script 產生它） | **27.51** | 10.50 | 62.8% |

`xgboost_metrics.parquet` 同時也被 `scripts/15_make_ml_diagnostic_figures.py` 拿來當 Fig 5 標題的數值來源，所以圖上是 RMSE=23.05；但若報告引用 `model_comparison.parquet`，就會出現「圖上 23.05、表格 27.51」的對不上現象。

### 可能原因

1. **最可能**：`model_comparison.parquet` 是當初手動建立的，commit `60aa1fb`（你把 XGBoost 從 `max_depth=6` 改成 `max_depth=3`、`learning_rate=0.03`）之後沒有重新產生，記錄的是舊版超參數的結果。
2. **次要**：兩個檔案 mtime 完全相同（都是 1779523162），但內容不一致 — 可能是 `model_comparison.parquet` 在某次 IDE / Jupyter 互動式階段被覆寫，但和 `10_train_xgboost.py` 用的不是同一個 model 物件。
3. **不可能但要排除**：兩支 script 都沒有控制 `random_state`，理論上不會出現這種大幅落差，但建議跑一次 `cat scripts/10_train_xgboost.py | grep random_state` 確認確實設定為 42。

### 建議修法

- **方案 A（推薦）**：刪掉 `model_comparison.parquet`，另寫 `scripts/10b_collate_model_comparison.py`，每次 build 自動讀 `xgboost_metrics.parquet` + `lasso_metrics.parquet` 拼起來。這樣 single source of truth。
- **方案 B**：把 `10_train_xgboost.py` 結尾改成同時更新 `model_comparison.parquet`（讀 lasso_metrics、append xgb、寫回去）。

---

## Issue #2 — XGBoost 整體 RMSE 比 Lasso 還差

**Severity**: 嚴重 / 必須在 report 解釋
**Status**: Open（需要文字 + 可能需要 robustness check）

### 敘述

目前的整體比較（以 `xgboost_metrics.parquet` 與 `lasso_metrics.parquet` 為準）：

| Model | RMSE | MAE | Directional Accuracy |
|---|---|---|---|
| **Lasso (alpha=1.0)** | **12.93** | **7.25** | 60.6% |
| **XGBoost** (depth=3, lr=0.03, n=300) | 23.05 | 9.19 | **64.8%** |

兩個指標互相矛盾：
- 量級指標（RMSE、MAE）：Lasso 大贏
- 方向指標（Directional Accuracy）：XGBoost 大贏

如果 report 直接寫「Lasso 比 XGBoost 好」會誤導讀者，因為這不是模型容量不足，而是兩個模型有不同的失敗模式。

### 可能原因

1. **最主要**：Lasso 因為 L1 正則化把多數係數壓縮接近 0，預測會偏保守、貼近樣本均值。在 protest_count 大多時間很低（每週個位數）但偶爾爆衝（黎以衝突、Imamoglu 事件）的長尾分布下，這種「保守預測」剛好避開了少數極端事件帶來的大誤差，所以 RMSE 看起來漂亮。
2. **XGBoost 反方向問題**：XGBoost 嘗試擬合「衝突訊號上升 → protest 上升」這個關係，但量級錯誤（見 Issue #3），少數樣本誤差被 RMSE 平方放大，整體 RMSE 被拉高。
3. **次要**：可能 Lasso 在標準化後對 multicollinearity（rolling × lag × diff 之間互相相關）處理得比較乾淨；XGBoost 對相關特徵會分散權重，反而在小樣本上不穩定（每國 ~470 train rows）。
4. **要排除**：請確認 `09_train_lasso.py` 跟 `10_train_xgboost.py` 用的是**完全一樣**的 `time_train_test_split`（同一份 `feature_matrix.parquet`、同樣 80/20 切點），否則上面這張比較表沒有意義。

### 建議在 report 的寫法（草稿）

> 在整體 RMSE 上，Lasso（12.93）優於 XGBoost（23.05），但這並非代表 Lasso 是更好的模型。Lasso 受 L1 收斂壓力影響，預測量級保守地往樣本均值靠攏；在 protest_count 呈現偶發性爆衝的長尾分布下，此「保守偏差」剛好避開了少數結構性事件所造成的大量誤差。相對地，XGBoost 雖然嘗試擬合衝突結構與抗議規模間的非線性關係，但因為對 Lebanon 與 Turkey 等具結構性斷裂的時期估計量級錯誤（見 Issue #3），被少數樣本拉高 RMSE。Directional Accuracy 上 XGBoost（64.8%）仍勝出，顯示其在「趨勢方向」上更可靠。實務上，這建議以 Lasso 作為短期 baseline、XGBoost 作為趨勢偵測模型，搭配使用。

### 建議行動

- [ ] 在 `report/ml.md` 寫出「為什麼 Lasso 看起來贏」的解釋（草稿如上）
- [ ] 加一個 robustness check：對 y 取 log1p 後再跑兩個模型比較 — 如果 log scale 下 XGBoost 反超 Lasso，就證實了上面的「長尾被 RMSE 懲罰」假設

---

## Issue #3 — XGBoost 對 Lebanon 系統性 over-predict、對 Turkey 系統性 under-predict

**Severity**: 核心發現 / 必寫進 report
**Status**: Open（已用 Fig 5/6 視覺化，等文字解讀）

### 敘述

Per-country test-set 指標（從 `xgboost_predictions.parquet` 重新拆解）：

| 國家 | n_test | RMSE | MAE | actual_max | predicted_max | 狀態 |
|---|---|---|---|---|---|---|
| Argentina (AR) | 112 | **4.65** | 3.04 | 31 | 19.7 | ✅ 正常 |
| Sri Lanka (CE) | 112 | **4.86** | 3.72 | 23 | 11.0 | ✅ 正常 |
| Chile (CI) | 112 | **3.91** | 2.90 | 20 | 16.9 | ✅ 正常 |
| Turkey (TU) | 112 | **22.62** | 12.21 | 197 | 60.4 | ⚠️ structural break under-predict |
| **Lebanon (LE)** | 112 | **45.65** | 24.06 | 64 | **220.9** | ❌ 系統性 over-predict |

整體 RMSE 23.05 幾乎全部由 LE + TU 兩國貢獻；AR/CE/CI 三國的模型其實表現得相當乾淨（RMSE 4-5）。

最嚴重的 5 個誤差：

| 國家 | 週次 | actual | predicted | 誤差 | 對應實際事件 |
|---|---|---|---|---|---|
| LE | 2024-11-25 | 22 | 220.9 | +198.9 | 黎以衝突升級期 |
| LE | 2024-09-30 | 38 | 209.0 | +170.9 | 同上 |
| **TU** | **2025-03-17** | **197** | **32.8** | **-164.2** | **Imamoglu 被捕大規模抗議** |
| LE | 2024-10-07 | 21 | 181.4 | +160.4 | 以色列入侵黎南 |
| LE | 2024-10-28 | 9 | 169.4 | +160.4 | 同上 |

新版 `figures/fig5_pred_vs_actual.png`、`figures/fig6_residual_plot.png` 已用 country 顏色重新繪製，紫色（LE）與橙色（TU）的異質性一眼可見。

### 可能原因

#### A. Lebanon over-prediction 的可能原因

1. **最可能 — Target / Feature 口徑不一致**：根據 `docs/summary_D.md`：
   - `protest_count` = 限定 `country_1 == country_2`（**只算國內抗議**）
   - `avg_tone`、`avg_goldstein`、`n_material_conf` = **不限國內**

   2024 年 9-11 月黎以衝突期間，goldstein 與 material_conf 因為跨國衝突暴衝；XGBoost 看到「衝突結構訊號 = 高」就推論「protest_count 應該也高」，但實際 protest_count 屬於國內抗議，並沒有跟著漲。

2. **次要 — 訓練集沒看過類似情境**：train 區間是 2015-02 到 2024-01（左右 80%），LE 在 train 期間沒有同等規模的跨境戰爭 → 模型沒學過「外部衝突 ≠ 國內 protest」的解耦關係。

3. **可能但要驗證 — Feature interaction `tone_goldstein_inter`**：B 在 commit `02897c9` 加入這個交互特徵後，可能放大了「tone 負 × goldstein 負」的訊號強度，對 LE 這種兩個都極端的情境特別敏感。

#### B. Turkey under-prediction 的可能原因

1. **最可能 — Structural break 不在 train 分布內**：Imamoglu 被捕（2025-03-19）是政治事件，本身是「衝擊」而非衝突訊號的延續累積。Train 期間 TU 的 protest_count 範圍大約是 0-50，模型上限被截斷，無法外推到 actual=197 的等級。
2. **次要 — Lag 訊號被平滑化**：模型用的是 lag 1-12 週與 rolling 4/8/12 週的特徵，這些對「瞬間爆發」的政治抗議反應較慢；當 lag 1 還沒看到 protest 訊號時，模型只能依賴 tone / goldstein，但這類政治事件不一定有顯著的事前媒體訊號累積。
3. **要排除 — 是不是只有 lag1 在訓練時被 leak**：請確認 `add_target` + `prepare_xy` 的 lag 處理是否完全不會洩漏未來資訊（特別是 `protest_count` 自身的 lag），這在 time-series ML 是最常見的 bug 來源。

### 建議在 report 的寫法（草稿）

> 將測試集 RMSE 拆解到各國後可以看見一個明顯的異質性：阿根廷、智利、斯里蘭卡三國的 RMSE 皆在 4-5 之間，但黎巴嫩高達 45.7、土耳其 22.6。整體 RMSE 23.05 幾乎全部由這兩個國家貢獻。
>
> 黎巴嫩的偏誤呈現系統性的 over-prediction：2024 年 9-11 月黎以衝突升級期間，模型多次將週抗議數估計到 150-220，但實際國內抗議事件僅為 9-38。這反映了本研究的一個資料口徑張力 — 目標變數 `protest_count` 限定於國內抗議（country_1 == country_2），但衝突結構特徵 `avg_goldstein`、`n_material_conf` 並未限定國內。當跨國武裝衝突大幅推升衝突結構訊號時，XGBoost 將之誤解讀為國內動員訊號。
>
> 土耳其則呈現相反方向的偏誤：2025 年 3 月伊斯坦堡市長 Imamoglu 被捕後，單週抗議數達 197，但模型僅預測 33。這類因突發政治事件導致的 structural break 不在訓練分布範圍內，且 lag / rolling 特徵對瞬時衝擊反應較慢，模型結構上難以外推。
>
> 兩種偏誤揭示了基於媒體訊號的抗議預測模型的本質限制：模型可以捕捉「衝突結構持續惡化 → 抗議逐步累積」的因果鏈，但對「跨境戰爭引發的訊號污染」與「政治衝擊型 structural break」皆缺乏外推能力。

### 建議行動

- [ ] **Robustness check（高優先）**：用 `protest_count_all`（不限 country_1==country_2）當 target 重跑 XGBoost。若 LE 的 RMSE 明顯下降，就證實「target/feature 口徑不一致」是主因，可在 report 直接引用。
- [ ] **可選**：在 train/test split 之外，再做一次 **leave-one-country-out** 驗證，看看模型對 LE / TU 是不是即使在 train 看過也學不起來。
- [ ] 在 `report/ml.md` 寫出上述偏誤解釋（草稿如上）並引用新版 Fig 5/6。

---

## 建議 commit message

```
docs: add issue board for ML pipeline (model comparison, RMSE inversion, per-country bias)
```
