# 成員 F 工作手冊

> 你負責**視覺化 + 報告整合**:Week 1 Day 1-3 已建好整個 repo 骨架(包含其他成員的程式框架),Week 1 Day 4-7 起逐步產出視覺化,Week 3 整合所有人章節成最終 report。

---

## 全部交付物總覽

| 週次 | 交付物 | 對誰交付 | 截止 |
|---|---|---|---|
| W1 Day 1-3 | repo 結構、`requirements.txt`、`src/{constants,viz_template,data_loader}.py`、其他成員 framework、tests、scripts、`report/` 骨架 | 全組 | **Day 1-3 已完成** |
| W1 Day 4-7 | Fig 1(雙軸折線)+ Fig 2(heatmap) | 整合用 | Day 7 結束 |
| W2 中 | Fig 3(UMAP)+ `report/` 各章節骨架(已備好) | 全組 | Week 2 中 |
| W2 末 | 整合 Fig 1-7(統一風格 review) | 全組 | Week 2 末 |
| W3 末 | 整合所有圖(含 Fig 8)+ `report/final.md` 排版 + 簡報視覺 | 全組 | Week 3 末 |

---

## 依賴關係

- **Day 1-3 不依賴任何人**(已完成)。
- **Day 4-7 依賴 A**:畫 Fig 1, 2 需要 `output/country_weekly.parquet`。
- **Week 2 中依賴 A**:畫 Fig 3 需要 `output/country_eventmix_weekly.parquet`。
- **Week 2-3 依賴所有人**:整合 Fig 4 (D)、Fig 5/6 (E)、Fig 7 (C)、Fig 8 (D 產出) + 各章節初稿。
- **依賴你的人**:全部成員(用你建好的 `src/`、`scripts/`、`viz_template`)。

---

## Week 1 Day 1-3:repo 結構(已完成)

> 此階段在你閱讀本文件時已完成。內容供你 review 並回頭修補。

### 已建立內容
- 目錄結構:`data/`、`output/{,arimax_results,ccf_granger,ml_results}/`、`figures/`、`src/`、`tests/`、`scripts/`、`report/`、`notebooks/{A_data_engineering,B_feature_engineering,C_arimax,D_statistics,E_machine_learning,F_visualization}/`
- `requirements.txt` + `.gitignore`(`.duckdb`、`.parquet` 都排除)
- `src/__init__.py`、`src/constants.py`、`src/viz_template.py`、`src/data_loader.py`
- 其他成員的 framework:`src/{feature_engineering,arimax_model,stat_tests,ml_models}.py`
- 測試:`tests/test_*.py`(4 個檔)
- 腳本:`scripts/01_*.py` ~ `scripts/16_*.py` + `sanity_check.py`
- 報告骨架:`report/{data,feature_engineering,methodology,statistics,ml,final}.md`

### Day 3 結束前的 checklist
- [ ] 已 push 全部上 main branch
- [ ] 群組通知所有人 clone + `pip install -r requirements.txt` + `pytest tests/`
- [ ] 確認 README.md 的 Google Drive 下載連結仍有效

---

## Week 1 Day 4-7:Fig 1 + Fig 2

### 你會用到的檔案
- `scripts/12_make_eda_figures.py`(已備好)

### 步驟

#### Step 1:確認 A 的 `country_weekly.parquet` 已到位
- 跑 `python scripts/sanity_check.py` 應顯示 [PASS] country_weekly

#### Step 2:跑 EDA figures
- 執行 `python scripts/12_make_eda_figures.py`
- 預期產出:
  - `figures/fig1_tone_vs_protest_per_country.png`(每國一個 panel,左 tone、右 protest)
  - `figures/fig2_tone_goldstein_heatmap.png`(行=國、列=週、色=值)

#### Step 3:檢查視覺品質
- 看 Fig 1 每國的趨勢是否合理(已知重大事件如 2022 斯里蘭卡崩潰應該看得到 protest 暴增)
- Fig 2 heatmap x 軸 tick 不要太密(已限制每季一個 tick)
- 若 tone 平均值 scale 不對(預期 [-5, 5]),回頭跟 A 確認 `weighted_tone` 算法

---

## Week 2:Fig 3 + 報告框架 + 整合 Fig 1-7

### Task 1:Fig 3(UMAP)
- 等 A 交 `country_eventmix_weekly.parquet`
- 執行 `python scripts/13_make_umap_figure.py`
- 預期產出 `figures/fig3_umap_eventmix.png`(每點 = 某國某週,顏色=國家)
- 若 5 國群聚不明顯,試調 `n_neighbors`、`min_dist` 參數

### Task 2:整合 Fig 4-7(收齊各成員產出)

| 圖 | 來源 | 你做什麼 |
|---|---|---|
| Fig 4 (CCF heatmap) | D 跑 `07_compute_ccf.py` 產出資料,你跑 `14_make_ccf_heatmap.py` 出圖 | 自動完成 |
| Fig 5, 6 (預測 vs 實際 / 殘差) | E 跑 `10_train_xgboost.py` 產出資料,跑 `15_make_ml_diagnostic_figures.py` 出圖 | 你 review 風格 |
| Fig 7 (ARIMAX forecast) | C 跑 `06_run_arimax.py` 產出資料,跑 `16_make_arimax_forecast_figure.py` 出圖 | 你 review 風格 |

風格 review 重點:
- 字型一致(Helvetica/Arial)
- 國家顏色一致(`COUNTRY_COLORS`)
- 標題格式一致(`apply_style()` 已統一)
- 軸標籤英文,大小寫一致

若不一致,直接修腳本參數 + 重跑 + commit。

### Task 3:報告框架(已備好)
- `report/{data,feature_engineering,methodology,statistics,ml,final}.md` 都已備骨架
- W2 中通知所有人開始填內容(已內含填寫指引)

---

## Week 3:整合 + 排版 + 簡報

### Task 1:Fig 8(SHAP)整合
- 等 D 跑完 `11_compute_shap.py`,確認 `figures/fig8_shap_summary.png` 已生成
- 若風格不一致(SHAP 預設用自己的色板),可在 notebook 重新出圖,套 `apply_style()`

### Task 2:`report/final.md` 整合
1. 把 5 個章節(`data.md`、`feature_engineering.md`、`methodology.md`、`statistics.md`、`ml.md`)的內容組進 `report/final.md`
2. 統一標題層級(`#` = report 標題、`##` = 大章、`###` = 小節)
3. 每章節重新編號圖(Fig 1, 2, ..., Table 1, 2, ...)
4. 統一引用格式(APA / IEEE 看老師要求)
5. 校稿(錯字、標點、格式一致)

### Task 3:輸出 PDF
建議用 pandoc:
```bash
# macOS:確認已裝 mactex 或 basictex
pandoc report/final.md -o report/final.pdf \
  --pdf-engine=xelatex \
  -V mainfont="PingFang TC" \
  -V geometry:margin=1in
```

若中文亂碼:確認 `mainfont` 是系統有的繁中字型(`fc-list :lang=zh-tw` 查)。

### Task 4:簡報視覺

幫各成員設計 slide 模板:
- 標題頁
- 章節分隔頁(Data / Methodology / Results / Discussion)
- 內容頁模板(標題 + 圖 + 1-2 個重點)
- Color scheme 跟報告一致(用 `src.constants.COUNTRY_COLORS`)

工具建議:
- **Keynote / PowerPoint**:手動排版
- **Marp** / **reveal.js**:Markdown 轉 slide(適合工程組)
- **Canva**:模板多

---

## 常見陷阱

| 陷阱 | 後果 | 解法 |
|---|---|---|
| 各成員自己選顏色 | 風格不一致 | 強制用 `from src.constants import COUNTRY_COLORS` |
| 圖片 dpi 太低 | 報告印出來糊 | `viz_template.save_fig` 已內建 dpi=150 |
| Heatmap x 軸 tick 太密 | 看不懂 | `12_make_eda_figures.py` 已限制每季一個 |
| 引用「Fig X」但檔名格式不一致 | 引用混亂 | 統一檔名 `figN_<short_name>.png` |
| 所有人 commit 圖到 repo | git repo 變大 | `.gitignore` 已排除 `figures/*`,改用 cloud share |
| 報告排版用 Word | markdown 反覆轉容易亂 | 一路用 markdown,最後 pandoc 出 PDF |
| 簡報跟報告風格不一致 | 答辯時觀眾混亂 | 簡報用同樣 color palette + 字型 |
| F 自己晚交圖 | 報告整體交不出來 | Day 4-7 就先把 Fig 1, 2 弄出來,後面只是整合 |

---

## 交付檢查清單

- [x] **W1 Day 3**:repo 結構 + `src/` + `tests/` + `scripts/` + `report/` 骨架 + `requirements.txt` + `.gitignore` 全部 push 上 main
- [ ] **W1 Day 7**:`figures/fig1_tone_vs_protest_per_country.png` + `figures/fig2_tone_goldstein_heatmap.png`
- [ ] **W2 中**:`figures/fig3_umap_eventmix.png`
- [ ] **W2 末**:Fig 4-7 風格統一 review + 各章節骨架已被填寫中
- [ ] **W3 末**:Fig 8 整合 + `report/final.pdf` + 簡報投影片
