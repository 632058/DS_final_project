# Issue Board — Member E (ML)

> Owner: Member E（曾弘舟）
> Status snapshot: 2026-05-23（updated after Issue #1 fix + data-path fix）
> Scope: `scripts/09_train_lasso.py`, `scripts/10_train_xgboost.py`, `scripts/10b_collate_model_comparison.py`, `scripts/11_compute_shap.py`, `output/ml_results/*`

## 進度總覽

| Issue | Status | 重點 |
|---|---|---|
| #1 `model_comparison.parquet` 對不上 | ✅ Resolved | 改成由 `scripts/10b_collate_model_comparison.py` 自動由 lasso/xgb metrics 拼出。 |
| #2 XGBoost 整體 RMSE 比 Lasso 還差 | ✅ Largely resolved | data path bug 修掉後 XGBoost RMSE 13.24 ≈ Lasso 12.48；MAE 上 XGBoost 反贏。 |
| #3 LE over-predict / TU under-predict | 🟡 Partial | LE 改善 62.7%（RMSE 45.65 → 17.03）；TU 幾乎沒變（22.62 → 22.87）。Task #3 regime+SHAP 診斷：LE 是 OOD 外推、TU 是模型結構缺 protest 自迴歸特徵（90 個特徵裡完全沒有 `protest_count_*_lag*`，但實測 lag-1 自相關 0.53）。修法：補上 protest_count_all 系列 lag/rolling 特徵。 |

下面三項是原始 issue 紀錄與後續更新。

---

## Issue #1 — `model_comparison.parquet` 與 `xgboost_metrics.parquet` 數字對不上

**Severity**: 嚴重 / 必修
**Status**: ✅ Resolved（2026-05-23）

### 修法

採用方案 A：新增 `scripts/10b_collate_model_comparison.py`，每次 build 自動從 `xgboost_metrics.parquet` + `lasso_metrics.parquet` 拼出 `model_comparison.parquet`，single source of truth。下次 build 流程：

```bash
python scripts/09_train_lasso.py
python scripts/10_train_xgboost.py
python scripts/10b_collate_model_comparison.py   # 一定要最後跑
```

### 原始敘述（保留）

`output/ml_results/` 底下兩個檔案對 XGBoost 報出來的指標不一致：`xgboost_metrics.parquet` 是 script 自動產出（commit `60aa1fb` 改超參後重產），`model_comparison.parquet` 是更早手動建立的舊版超參結果，沒人重產。`scripts/15_make_ml_diagnostic_figures.py` 用的是 `xgboost_metrics.parquet`，所以圖跟表會對不上。

---

## Issue #2 — XGBoost 整體 RMSE 比 Lasso 還差

**Severity**: 原本「嚴重 / 必須在 report 解釋」
**Status**: ✅ Largely resolved（2026-05-23，由 data-path fix 順帶解決）

### 後續發現

問題並非模型本身有缺陷，而是 **data path bug** —— `src/data_loader.load_country_weekly()` 原本讀 `data/country_weekly.parquet`（5/19 由 `01b_build_country_weekly_from_parquet.py` 用 `country_1 = target country` 聚合），但 Member A 在 5/21 重寫的 `01_build_country_weekly.py` 改用 `ActionGeo_CountryCode = target country` 聚合並寫到 `output/country_weekly.parquet`。整條 feature → ML pipeline 一直用舊聚合版本。

### Path fix 後（target 顯式設為 `protest_count_all`）的新指標

| Model | RMSE | MAE | Directional Accuracy |
|---|---|---|---|
| Lasso (alpha=1.0) | **12.48** | 6.88 | 64.2% |
| XGBoost (depth=3, lr=0.03, n=300) | 13.24 | **6.84** | **65.5%** |

兩個模型差距已經很小：RMSE 上 Lasso 微贏 0.76，MAE 上 XGBoost 微贏，Dir Acc 上 XGBoost 微贏。原本「Lasso 大贏 XGBoost」的命題不成立。

### Report 寫法建議

> XGBoost 與 Lasso 在 hold-out 測試集上表現非常接近（RMSE 13.24 vs 12.48；MAE 6.84 vs 6.88；Directional Accuracy 65.5% vs 64.2%）。Lasso 在 RMSE 上略勝，XGBoost 在 MAE 與方向預測上略勝。兩者實質效力相當，可視為互補：Lasso 提供穩定的線性基準與可解釋係數，XGBoost 捕捉非線性互動與趨勢方向。

---

## Issue #3 — XGBoost 對 Lebanon 系統性 over-predict、對 Turkey 系統性 under-predict

**Severity**: 核心發現 / 必寫進 report
**Status**: 🟡 Partial — LE 大幅收斂、TU 仍開放

### 原始假設（已證偽）

原本診斷為「target/feature 口徑不一致」（`protest_count` 限定 `country_1 == country_2`、其它特徵不限）。實際盤點後：

- `data/country_weekly.parquet`（pipeline 在讀的舊檔）的 `protest_count` 是 01b 用 `country_1 = target` 聚合，**並非** domestic-only
- `output/country_weekly.parquet`（新檔）才有真正 domestic-only 的 `protest_count` 跟全事件的 `protest_count_all`
- 修 path bug 之前，features 跟 target 其實**都是**「country 為事件來源」這個 scope，沒有口徑不一致

所以「scope mismatch」這個 hypothesis 在事實面是錯的。

### Path fix 後的 per-country test 指標

| 國家 | OLD RMSE | NEW RMSE | OLD pred_max | NEW pred_max | 狀態 |
|---|---|---|---|---|---|
| Argentina (AR) | 4.65 | 4.72 | 19.7 | 21.3 | ✅ 正常 |
| Sri Lanka (CE) | 4.86 | 5.11 | 11.0 | 12.1 | ✅ 正常 |
| Chile (CI) | 3.91 | 3.79 | 16.9 | 19.5 | ✅ 正常 |
| **Lebanon (LE)** | **45.65 → 17.03** | (↓ 62.7%) | **220.9 → 85.6** | | 🟡 殘留 over-predict 但量級大幅縮小 |
| **Turkey (TU)** | 22.62 | 22.87 | 60.4 | 61.5 | ❌ 幾乎沒變，Imamoglu 衝擊仍主導 |

LE 的進步來自於新聚合口徑（`ActionGeo` vs `country_1`）— 以黎衝突期間大量事件 `country_1 = 以色列`、`ActionGeo = LE`，新口徑把這些事件正確算入 LE 的脈絡，模型不再用「LE 是主動方」的偏狹視角去學。

### Path fix 後最嚴重 5 個殘差

| 國家 | 週次 | actual | predicted | 誤差 | 對應實際事件 |
|---|---|---|---|---|---|
| **TU** | 2025-03-17 | **197** | 34.0 | **−163.0** | Imamoglu 被捕大規模抗議 |
| TU | 2025-03-10 | 99 | 10.8 | −88.2 | Imamoglu 前一週 |
| LE | 2026-03-02 | 14 | 85.6 | +71.6 | 殘留 over-predict |
| TU | 2025-09-22 | 69 | 13.1 | −55.9 | TU 後續事件 |
| LE | 2024-11-25 | 22 | 77.0 | +55.0 | 黎以衝突期殘留 |

### Task #3 診斷結果（2026-05-23）：regime check + SHAP

#### Regime check — LE 與 TU 的本質完全不同

| 特徵 | TU train_max | TU test_max | TU OOD? | LE train_max | LE test_max | LE OOD? |
|---|---|---|---|---|---|---|
| `n_material_conf` | 6622 | 784 | ❌ | 1427 | **3630** | ✅ |
| `n_material_conf_rolling_mean_4w` | 4000 | 497 | ❌ | 1098 | **2578** | ✅ |
| `n_material_conf_lag1` | 6622 | 784 | ❌ | 1427 | **3630** | ✅ |
| `tone_goldstein_inter` | 8.38 | 2.19 | ❌ | 21.78 | **25.40** | ✅ |
| `n_verbal_conf` | 3885 | 634 | ❌ | 1083 | **1593** | ✅ |
| `material_conf_ratio` | 0.33 | 0.24 | ❌ | 0.39 | **0.48** | ✅ |

**結論**：
- **LE 是真正的 OOD 外推問題**：2024-09 之後黎以衝突的衝突訊號全面超出 train 期間 max（2-3 倍）。樹模型對未見過的範圍只能用最後一個 split threshold 對應的葉節點推論 → 系統性 over-predict。
- **TU 並非 OOD 問題**：所有 test features 都在 train 範圍內（甚至遠遠低於 train max，因為 2024 年之前的 TU 衝突活動更大）。**Imamoglu 預測失準不是「沒見過這個訊號量級」**。

#### SHAP 證據 — TU 的真正瓶頸

| 殘差列 | 第一名 SHAP 特徵 | 值 | SHAP | 解讀 |
|---|---|---|---|---|
| TU 2025-03-17 (err=+163) | `protest_ratio` | 0.041 | **+14.2** | 模型唯一抓到的訊號是「當週 protest 比率已升高」，但只貢獻 +14 → 模型沒辦法把這個訊號放大 |
| TU 2025-03-10 (err=+88) | `protest_ratio` | 0.004 | −2.3 | 預測下週 protest=99，但當週 protest 還沒發生 → 零訊號可用 |
| LE 2026-03-02 (err=−72) | `n_material_conf_diff_1w` | 1437 (OOD) | **+34.3** | 訓練從未看過這麼大的 diff，樹外推到 +34 |
| LE 2024-11-25 (err=−55) | `n_verbal_conf` | 846 (OOD) | +10.5 | 同上，n_material_conf_lag10=1584 也是 OOD |

**最關鍵發現** — 整份 feature_matrix 的 90 個欄位裡，**完全沒有 `protest_count_*_lag*` 或 `protest_ratio_lag*` 特徵**。lag features 只覆蓋 `avg_tone`、`avg_goldstein`、`n_material_conf`。

但實測 TU train 期間 `protest_count_all` 的 lag-1 自相關 = **0.531**（強訊號）。模型對抗議的自迴歸訊號是完全瞎的。

#### 對 Imamoglu 序列的反事實檢驗

```
週次         protest_count_all  protest_count  protest_ratio
2025-02-17                  8              4         0.004
2025-02-24                 10              4         0.003
2025-03-03                  6              3         0.003
2025-03-10                  7              0         0.004    ← 預測下週 actual=99，模型沒任何訊號
2025-03-17                 99             88         0.041    ← 預測下週 actual=197，模型只從 protest_ratio 抓到 +14
2025-03-24                197            144         0.066
2025-03-31                 71             44         0.039
```

**Row 2025-03-17 的 target = 2025-03-24 的 protest_count_all = 197**。這一列如果有 `protest_count_all`（當週=99）做為 feature，模型就能用簡單的 AR(1) 邏輯（自相關 0.53）推到下週應該也 100+，再加上現有 protest_ratio 放大，預測可能落到 150-200 範圍。

**Row 2025-03-10 的 target = 2025-03-17 的 99** — 這一列當週 protest_count_all=7（完全正常），這是真正的 unpredictable shock，任何模型都打不到。

### 殘留問題的修正版假設（基於上述證據）

#### A. LE 殘留 over-predict — 確認為 OOD 外推

樹模型在 LE 訓練分布外（2024-09 黎以衝突期間衝突訊號量級史上新高）只能用 saturate 的 leaf prediction。
- **次要因素**：`tone_goldstein_inter` 在 SHAP 中對 LE 兩個殘差都是 −4 左右（不是主要 driver，原本的懷疑撤回）
- **主要因素**：`n_material_conf_diff_1w`、`n_material_conf_lag10`、`n_verbal_conf` 等變數的 OOD 值集中推高預測

#### B. TU under-predict — **重新定性為「缺乏自迴歸特徵」而非「structural break」**

原本診斷說「structural break 不在 train 分布內」是錯的 — features 都在 train 分布內。真正的問題是模型沒有 protest_count 系列的 lag 特徵，所以即使本週 protest 已經暴升，模型也無法把這個訊號傳遞到下週預測。

對於 Imamoglu 完全突發的第一週（2025-03-10 → 預測 2025-03-17=99）仍然無解 — 那是真正的「零訊號 shock」，任何純媒體訊號模型都打不到。

### 建議改善行動（按優先順序）

#### 🔴 高優先 — 加 protest 自迴歸特徵（預期能解掉 TU 殘差大宗，並順帶緩解 LE）

在 `src/feature_engineering.py` 加入 `protest_count_all` 系列的 lag/rolling/diff，與現有 tone/goldstein/material_conf 的處理一致：

```python
# 在 ROLLING_LAG_TARGETS 加上 protest_count_all
ROLLING_LAG_TARGETS = ['avg_tone', 'avg_goldstein', 'n_material_conf', 'protest_count_all']
```

並從 `src/ml_models.NON_FEATURE_COLS` 移除 `'protest_count_all'`（讓當週值也能當 feature；target 是 `shift(-1)`，無 leakage）。

**預期效果**：
- TU 2025-03-17（預測下週 197）：當週 `protest_count_all=99` 直接作為強訊號 + 同樣強的 lag-1 自相關 → 預測有機會跳到 100-200 範圍，殘差大幅縮小
- LE 2026-03-02（OOD 過度預測）：當週 `protest_count_all=2`（極低）給模型負向訊號，能對沖部分 OOD 衝突訊號帶來的虛假 over-predict

成本：~10 行程式碼 + 重跑 04→09→10→10b→11→15 pipeline。

#### 🟡 中優先 — 對 LE OOD 問題的後續處理

A.1 **Winsorize features at train p99**：對 `n_material_conf`、`n_verbal_conf`、`n_material_conf_lag*` 在 inference 時 clip 到 train 的 p99，避免樹外推。代價：實作上需要在 `prepare_xy` 階段加 clip，且要記住 train p99。

A.2 **加 domestic-only 衝突特徵**：跟 Member A 協調，請 `01_build_country_weekly.py` 多輸出 `n_material_conf_domestic`、`avg_goldstein_domestic`（限定 `Actor1CountryCode=Actor2CountryCode=ActionGeo`）。讓模型同時看到「整體衝突」與「國內衝突」兩家族，學會在 LE 跨境戰爭期間哪一家才該主導 protest 預測。

#### 🟢 低優先 — Imamoglu 第一週（2025-03-10 → 預測 99）

這是真正的零訊號 shock，任何純媒體模型結構上打不到。建議 report 中坦承為**本研究方法的本質限制**，不要試圖過度工程化去硬擬合。可考慮：

- 預測同時輸出 quantile interval（XGBoost quantile loss）顯示不確定性
- 在 report 寫法中明確區分「可預測的延續期（2025-03-24 的 197）」與「不可預測的衝擊起點（2025-03-17 的 99）」

### Report 寫法（草稿，已更新數字 + Task #3 診斷）

> 將測試集 RMSE 拆解到各國後可以看見一個明顯的異質性：阿根廷、智利、斯里蘭卡三國的 RMSE 皆在 4-5 之間，黎巴嫩 17.0，土耳其 22.9。整體 RMSE 13.2 的主要殘留集中在這兩個國家，但成因截然不同。
>
> 土耳其的最大殘差（2025-03-24 actual=197、predicted=34）對應 Imamoglu 被捕後的大規模抗議。Regime check 顯示該週所有 conflict 訊號都在訓練分布內，並非「沒見過這個量級的衝突」。SHAP 分析揭示真正的瓶頸：本模型 90 個特徵中完全沒有 protest 自迴歸特徵（`protest_count_*_lag*`），即使前一週抗議數已從 7 跳到 99，模型也無法把這個訊號傳遞到下週預測。實測 TU 訓練期間抗議數的 lag-1 自相關為 0.53（強訊號），這是現行模型結構上的盲點。
>
> 黎巴嫩在 2024 年 9-11 月黎以衝突升級期間仍呈現殘留的 over-prediction（誤差 50-70），但相較於原始 pipeline（最大誤差 +198）已大幅收斂，改善來源是 GDELT 事件聚合口徑從「主動方國家」改為「行為發生地」。殘留部分為真正的 out-of-distribution 外推 — 該段期間黎巴嫩的 `n_material_conf` 等衝突訊號達到訓練期間 max 的 2-3 倍，樹模型只能用最後一個 split 對應的葉節點外推。

---

## 建議 commit messages

```
feat: add 10b_collate_model_comparison.py to unify ML metrics source of truth
fix: load_country_weekly reads OUTPUT_DIR; use protest_count_all as ML target
docs: update issues_E.md with Issue #1/#2 resolution and Issue #3 reframe
docs: add Task #3 diagnosis to issues_E.md (regime + SHAP for LE/TU)
```
