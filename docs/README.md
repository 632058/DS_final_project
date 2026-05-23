# 期末專案組員工作手冊

> **最後更新**:2026-05-07
> **專案**:基於 GDELT 全球事件資料庫之國際事件趨勢分析與預測
> **時程**:3 週壓縮並行版(詳見 [`../PROJECT_PROPOSAL.md`](../PROJECT_PROPOSAL.md) 第 9 節)

每位組員一份手冊,寫清楚**每週要做什麼、用哪些檔案、產出什麼、交給誰**。

---

## 文件導覽

| 成員 | 工作主題 | 文件 |
|---|---|---|
| A | 資料聚合(三張表) | [member_A.md](member_A.md) |
| B | 特徵工程 | [member_B.md](member_B.md) |
| C | 時序建模 ARIMAX | [member_C.md](member_C.md) |
| D | 統計檢定 CCF + Granger | [member_D.md](member_D.md) |
| E | 機器學習 XGBoost + SHAP | [member_E.md](member_E.md) |
| F | 視覺化 + 報告整合 | [member_F.md](member_F.md) |
| Cross-cutting | Feature scope cleanup + Fig 5/6 改善計畫 | [feature_scope_and_fig56_plan.md](feature_scope_and_fig56_plan.md) |

---

## 整體流程依賴圖

```
Week 1 Day 1-3  全員平行起步:
                A: country_weekly (Day 3 必交,critical path)
                B: 熟悉 src/feature_engineering.py
                C: 熟悉 src/arimax_model.py
                D: 熟悉 src/stat_tests.py
                E: 熟悉 src/ml_models.py
                F: repo 結構已建好,熟悉 src/{constants,viz_template,data_loader}.py

Week 1 Day 4-7  跑得到資料就開工:
                A: country_monthly + schema doc
                B: 跑 04_build_feature_matrix.py
                C: 跑 05_run_adf.py (ADF 平穩性檢定)
                D: 跑 07_compute_ccf.py (初步 CCF)
                E: 跑 09_train_lasso.py (baseline)
                F: 跑 12_make_eda_figures.py (Fig 1, 2)

Week 2          核心並行:
                A: country_eventmix_weekly + Data 章節 + 協助 C 跑 ARIMAX (2-3 國)
                B: feature_matrix v2 (補強) + 支援 C/D/E
                C: 跑 06_run_arimax.py 全 5 國 + Fig 7
                D: 跑 08_run_granger.py + Fig 4 + 協助 E 跑 SHAP
                E: 跑 10_train_xgboost.py + Fig 5, 6
                F: 跑 13_make_umap_figure.py (Fig 3) + 報告框架 + 整合 Fig 1-7

Week 3          收斂:
                全員: 章節最終版 + 簡報 + reproducibility check
                F: 整合所有圖 + 報告排版 + 簡報視覺
```

**Critical path**:A 的 `country_weekly` 必須 Week 1 Day 3 結束前交付。

---

## 已建立的專案結構

F 已在 Week 1 Day 1-3 建立以下骨架,所有人 clone 後可直接使用:

```
DS_final_project/
├── data/                           # 原始 .duckdb 放這(.gitignore)
├── output/                         # 中間產物(.gitignore)
│   ├── arimax_results/             # C 產出
│   ├── ccf_granger/                # D 產出
│   └── ml_results/                 # E 產出
├── src/                            # 共用程式 (已備好)
│   ├── constants.py                # FIPS_COUNTRIES、路徑、顏色等共用常數
│   ├── viz_template.py             # 統一 matplotlib 風格
│   ├── data_loader.py              # load_country_weekly() 等
│   ├── feature_engineering.py      # B 用
│   ├── arimax_model.py             # C 用
│   ├── stat_tests.py               # D 用
│   └── ml_models.py                # E 用
├── scripts/                        # 一鍵執行腳本 (已備好)
│   ├── 01_build_country_weekly.py  # A
│   ├── 02_build_country_monthly.py # A
│   ├── 03_build_country_eventmix_weekly.py  # A
│   ├── 04_build_feature_matrix.py  # B
│   ├── 05_run_adf.py               # C
│   ├── 06_run_arimax.py            # C
│   ├── 07_compute_ccf.py           # D
│   ├── 08_run_granger.py           # D
│   ├── 09_train_lasso.py           # E
│   ├── 10_train_xgboost.py         # E
│   ├── 11_compute_shap.py          # D/E
│   ├── 12_make_eda_figures.py      # F
│   ├── 13_make_umap_figure.py      # F
│   ├── 14_make_ccf_heatmap.py      # F
│   ├── 15_make_ml_diagnostic_figures.py  # F
│   ├── 16_make_arimax_forecast_figure.py # F
│   └── sanity_check.py             # 檢查 parquet 輸出
├── tests/                          # pytest 單元測試 (已備好)
├── notebooks/                      # 個人實驗區(每人一個子資料夾)
├── figures/                        # 最終圖檔
├── report/                         # 各章節 markdown(已備骨架)
├── docs/                           # 本資料夾
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── PROJECT_PROPOSAL.md
└── README.md
```

> **重要**:`src/` 與 `scripts/` 已備好,直接用即可。如需擴充再修改原檔。

---

## 環境設定(每位組員一次性)

```bash
# 1. clone repo
git clone <repo-url>
cd DS_final_project

# 2. 建立虛擬環境
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. 安裝依賴
pip install -r requirements.txt
pip install -e .                  # 讓 scripts/ 能 import src/

# 4. 把 .duckdb 檔案放到 data/(從 README.md 的 Google Drive 下載)
# 預期路徑: data/gdelt_filtered_20251001_20260428.duckdb

# 5. 跑單元測試確認環境 OK
pytest tests/

# 6. 跑 sanity check(此時還沒有資料,顯示 SKIP 正常)
python scripts/sanity_check.py
```

---

## 跨成員協議

### 中間檔案統一格式
- 全部用 **Parquet**(不是 CSV)
- 命名:`output/<table_name>.parquet`,讀取用 `src.data_loader.load_*()`

### 共用常數
- 一律從 `src.constants` import,不要 hardcode 國碼或路徑
- 例:`from src.constants import FIPS_COUNTRIES, COUNTRY_COLORS`

### 視覺化風格
- 所有圖用 `from src.viz_template import apply_style, save_fig`
- 在每個畫圖 script 開頭呼叫 `apply_style()` 一次
- 國家顏色:從 `COUNTRY_COLORS` 取
- 存檔:用 `save_fig(fig, 'figN_short_name')`,自動存到 `figures/`

### 圖表編號(對應 PROJECT_PROPOSAL §7.3.4)

| Fig | 內容 | 產出者 | 腳本 |
|---|---|---|---|
| 1 | 雙軸折線圖(tone vs protest) | F | `12_make_eda_figures.py` |
| 2 | tone/goldstein heatmap | F | `12_make_eda_figures.py` |
| 3 | UMAP event mix | F | `13_make_umap_figure.py` |
| 4 | CCF heatmap | D(資料)+ F(圖) | `14_make_ccf_heatmap.py` |
| 5 | Pred vs Actual 散點 | E | `15_make_ml_diagnostic_figures.py` |
| 6 | Residual plot | E | `15_make_ml_diagnostic_figures.py` |
| 7 | ARIMAX forecast | C | `16_make_arimax_forecast_figure.py` |
| 8 | SHAP summary | D/E | `11_compute_shap.py` |

### 命名規範
- 檔名:`lowercase_with_underscores.py`
- 變數名:`snake_case`
- DataFrame:`df_<目的>`
- 程式註解一律用 **英文**

---

## 同步會議

| 時點 | 確認事項 |
|---|---|
| W1 Day 3 結束 | A 已交 `country_weekly`;其餘人單元測試 pass |
| W1 Day 7 結束 | `feature_matrix` v1、Lasso baseline、Fig 1/2 已產出 |
| W2 中 | 所有模型可跑、所有圖可生成 |
| W2 末 | 各章節初稿完成 |
| W3 中 | 章節最終版 + 開始排版 + 簡報製作 |
| W3 末 | 最終交付 |

---

## 遇到問題的求助順序

1. 看自己的 `member_X.md`(常見陷阱、檢查清單)
2. 看 [`../PROJECT_PROPOSAL.md`](../PROJECT_PROPOSAL.md) 對應章節(技術原則、分析流程)
3. 直接看 `src/<相關檔案>.py` 的 docstring 與註解
4. 問 AI(請帶上專案 context:5 國 FIPS、`DATEADDED` int64、不能 shuffle)
