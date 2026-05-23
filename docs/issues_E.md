# Issue Board — Member E (ML)

> Owner: Member E（曾弘舟）
> Status snapshot: 2026-05-23（updated after Issue #1 fix + data-path fix）
> Scope: `scripts/09_train_lasso.py`, `scripts/10_train_xgboost.py`, `scripts/10b_collate_model_comparison.py`, `scripts/11_compute_shap.py`, `output/ml_results/*`

## 進度總覽

| Issue | Status | 重點 |
|---|---|---|
| #1 `model_comparison.parquet` 對不上 | ✅ Resolved | 改成由 `scripts/10b_collate_model_comparison.py` 自動由 lasso/xgb metrics 拼出。 |
| #2 XGBoost 整體 RMSE 比 Lasso 還差 | ✅ Largely resolved | data path bug 修掉後 XGBoost RMSE 13.24 ≈ Lasso 12.48；MAE 上 XGBoost 反贏。 |
| #3 LE over-predict / TU under-predict | 🟡 Partial | LE 改善 62.7%（RMSE 45.65 → 17.03）；TU 幾乎沒變（22.62 → 22.87），Imamoglu 結構性衝擊仍是主要殘留問題。原本「scope mismatch」假設**錯誤**，真正原因是 stale data + 不同國家事件聚合口徑。 |

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

### 殘留問題的假設

#### A. LE 還是有 over-predict（量級小很多）

1. **可能 — `tone_goldstein_inter` 在極端值放大訊號**：B 在 commit `02897c9` 加入這個交互特徵，對「tone 負 × goldstein 負」極端組合敏感；2024 黎以衝突期、2026-03 都是兩者極端負，可能仍被放大。
2. **可能 — Lag features 撞訓練分布外推極限**：12 週 lag + 4/8/12 週 rolling 在衝突高峰期全部 saturate，模型只能往上推。

#### B. TU under-predict（沒改善）

1. **最主要 — Structural break 不在 train 分布內**：Imamoglu 被捕（2025-03-19）是政治衝擊事件，本身是「shock」而非衝突訊號累積。Train 期間 TU 的 protest 量級被截斷在較低範圍，模型上限被限制，無法外推到 actual=197 的等級。
2. **次要 — Lag 訊號被平滑化**：lag 1-12 週與 rolling 4/8/12 週對「瞬間爆發」的政治抗議反應較慢；當 lag 1 還沒看到 protest 訊號時，模型只能依賴 tone / goldstein，但這類政治事件不一定有顯著的事前媒體訊號累積。

### 建議行動

- [ ] **Regime check（高優先）**：對 LE / TU 看 train vs test 期間 `n_material_conf`、`goldstein` 等關鍵特徵的分布，特別是 test 期極端值是否超出 train 的 max。若超出，就確認「extrapolation beyond training distribution」是主因。
- [ ] **SHAP 拆解最嚴重 5 個殘差**：用現有 `shap_values.npy` 抽出 TU 那 2 列、LE 那 2 列，看哪些 feature 在貢獻。如果 `tone_goldstein_inter` 或 `n_material_conf_lag*` 佔主導 → 證實假設。
- [ ] **Report 寫法**：把 TU Imamoglu 案例當成 media-signal-based 抗議預測模型的**本質限制**寫出來 — 模型可以捕捉「衝突結構持續惡化 → 抗議逐步累積」的因果鏈，但對「政治衝擊型 structural break」缺乏外推能力。

### Report 寫法（草稿，已更新數字）

> 將測試集 RMSE 拆解到各國後可以看見一個明顯的異質性：阿根廷、智利、斯里蘭卡三國的 RMSE 皆在 4-5 之間，黎巴嫩 17.0，土耳其 22.9。整體 RMSE 13.2 的主要殘留來自土耳其單一事件 — 2025 年 3 月伊斯坦堡市長 Imamoglu 被捕後的單週抗議數達 197，但模型僅預測 34。這類因突發政治事件導致的 structural break 不在訓練分布範圍內，且 lag / rolling 特徵對瞬時衝擊反應較慢，模型結構上難以外推。
>
> 黎巴嫩在 2024 年 9-11 月黎以衝突升級期間仍呈現殘留的 over-prediction（誤差 50-70），但相較於原始 pipeline（最大誤差 +198）已大幅收斂。改善來源是 GDELT 事件的「行為發生地」聚合口徑（`ActionGeo_CountryCode`）取代「主動方國家」（`country_1`），使得跨境衝突事件能正確歸入黎巴嫩脈絡而非以色列脈絡。

---

## 建議 commit messages

```
feat: add 10b_collate_model_comparison.py to unify ML metrics source of truth
fix: load_country_weekly reads OUTPUT_DIR; use protest_count_all as ML target
docs: update issues_E.md with Issue #1/#2 resolution and Issue #3 reframe
```
