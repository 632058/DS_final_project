# 成員 A 工作手冊

> 你負責**資料聚合**:把 70GB 的 GDELT raw events 變成下游可用的三張小表。**你是 critical path 的源頭** — Day 1-3 必須交付 `country_weekly`,B/C/D/F 都在等。

---

## 全部交付物總覽

| 週次 | 交付物 | 對誰交付 | 截止 |
|---|---|---|---|
| W1 Day 1-3 | `output/country_weekly.parquet` | B、F、C、D | **Day 3 結束(優先!)** |
| W1 Day 4-7 | `output/country_monthly.parquet` + `output/schema.md` | 全組 | Day 7 結束 |
| W2 中 | `output/country_eventmix_weekly.parquet` | F | Week 2 中 |
| W2 末 | 協助 C 跑 ARIMAX(2-3 國) | C | Week 2 末 |
| W2 末 | `report/data.md` 初稿 | F | Week 2 末 |
| W3 末 | `report/data.md` 最終版 + reproducibility check + `setup.md` | 全組 | Week 3 末 |

---

## 依賴關係

- **你不依賴任何人**,可以立刻開工。
- **依賴你的人**:B、F、C、D 從 Day 4 起;E 透過 B 拿 `feature_matrix`。
- **延遲影響**:你慢一天,整個專案順延一天。

---

## Week 1 Day 1-3:`country_weekly` 優先交付

### 你會用到的檔案
- `scripts/01_build_country_weekly.py`(已備好,直接執行)
- `scripts/sanity_check.py`(檢查產出)

### 步驟

#### Step 1:環境準備
1. 從 README.md 的 Google Drive 下載 `gdelt_filtered_20251001_20260428.duckdb`
2. 放到 `data/` 資料夾(專案根目錄已建好 `data/`)
3. 確認虛擬環境已啟動、依賴已安裝(見 `docs/README.md` 環境設定)

#### Step 2:基本連線測試
- 開個 Jupyter notebook 在 `notebooks/A_data_engineering/` 下
- 連 DuckDB,確認以下事項:
  - 總筆數、最早與最新 `DATEADDED`
  - 5 國(CE、AR、CI、TU、LE)各自筆數應 > 5000

> ⚠ **若某國資料 < 1000 筆,立刻通知組長** — 可能要替換國家。

#### Step 3:執行 country_weekly 產出腳本
- 在專案根目錄執行:`python scripts/01_build_country_weekly.py`
- 預期輸出:`output/country_weekly.parquet`,並印出每國週數
- 預期每國週數約 430(2018-02 至 2026-04)

#### Step 4:Sanity check(必做!)
- 執行 `python scripts/sanity_check.py`
- 確認所有 [PASS] 都顯示
- 任何 [FAIL] → 找出原因再交付,**不要把錯誤資料丟給下游**

#### Step 5:交付 + 通知
- 把 `output/country_weekly.parquet` 上傳到組內共用 cloud(因 .gitignore 排除)
- 群組通知:
  > @B @F @C @D `country_weekly.parquet` 已產出,共 [n] 列 × [k] 欄,涵蓋 5 國從 [start] 到 [end]。下游可以開工了。

---

## Week 1 Day 4-7:`country_monthly` + Schema doc

### 你會用到的檔案
- `scripts/02_build_country_monthly.py`(已備好)

### 步驟

#### Step 1:跑 monthly 聚合
- 執行:`python scripts/02_build_country_monthly.py`
- 產出:`output/country_monthly.parquet`

#### Step 2:撰寫 `output/schema.md`
- 對三張表(weekly、monthly、eventmix_weekly)各寫一段
- 每張表列:欄位名、型別、意義、範例值、備註
- 標明:weighted_*、count_*、average_* 三類欄位的差異
- 用 `pd.read_parquet(...).dtypes` 取得型別

---

## Week 2:eventmix_weekly + Data 章節 + 協助 C

### Task 1:跑 eventmix_weekly
- 執行:`python scripts/03_build_country_eventmix_weekly.py`
- 產出:`output/country_eventmix_weekly.parquet`(F 用來畫 Fig 3)
- 結構:每國每週 + 20 個 `ratio_<root_code>` 欄位

### Task 2:協助 C 跑 ARIMAX(負責 2-3 國)
- 跟 C 約好分工(建議:你跑 CE、AR、CI;C 跑 TU、LE)
- 執行:`python scripts/06_run_arimax.py CE AR CI`
- 產出在 `output/arimax_results/`(每國一個 `.json` + `_forecast.parquet`)
- 跑完通知 C 你負責的國家結果

### Task 3:撰寫 `report/data.md` 初稿(約 1500 字)

骨架已在 `report/data.md`,填內容:
1. **資料來源**:GDELT 2.0 介紹、時間範圍、表結構
2. **方法論決策**:為何用 `DATEADDED` 不用 `SQLDATE`(look-ahead bias)
3. **國碼系統**:FIPS 10-4 介紹、5 國代碼選擇理由
4. **聚合表設計**:三張表的 schema 與設計動機
5. **資料品質檢查**:missing rate、各國週數、極端值
6. **描述性統計**:5 國 protest_count / avg_tone / avg_goldstein 平均、std,引用 Fig 1, 2

---

## Week 3:Data 章節最終版 + Reproducibility check

### Reproducibility check
1. 開乾淨虛擬環境,從 git clone 開始
2. 按 `setup.md` 的順序執行所有 `scripts/01-03_*.py`
3. 比對 `country_weekly.parquet` 的 hash 跟 Week 1 交付的版本是否一致
   - macOS:`shasum -a 256 output/country_weekly.parquet`

### 撰寫 `setup.md`
- 從 git clone → 下載 .duckdb → pip install → 執行 scripts → 產出所有 parquet 的完整指令
- 所有人 reproducibility check 都靠這份

---

## 常見陷阱

| 陷阱 | 後果 | 解法 |
|---|---|---|
| `DATEADDED` 直接當 timestamp | SQL 跑錯 | 用 `LEFT(CAST(... AS VARCHAR), 8)` 再 `strptime` |
| 用 ISO 國碼('LKA') | GDELT 用 FIPS('CE'),會撈不到資料 | 看 `src/constants.py` 的 `FIPS_COUNTRIES` |
| `EventRootCode = 14`(int 比較) | 它是 varchar,要寫 `'14'` | SQL 已備好,用就對了 |
| 整檔讀進 pandas 再 GROUP BY | 70GB 會爆 RAM | 用 `scripts/01-03_*.py`(SQL 在 DuckDB 裡 GROUP BY 完才轉 DataFrame) |
| 存 CSV 給下游 | 型別會跑掉 | 一律用 Parquet |
| 沒做 sanity check 就交付 | 下游 debug 你的錯 | 一定跑 `python scripts/sanity_check.py` |

---

## 交付檢查清單

- [ ] **W1 Day 3**:`output/country_weekly.parquet` + sanity check 全 [PASS] + 通知 B/F/C/D
- [ ] **W1 Day 7**:`output/country_monthly.parquet` + `output/schema.md`
- [ ] **W2 中**:`output/country_eventmix_weekly.parquet`
- [ ] **W2 末**:協助 C 跑 ARIMAX(2-3 國,結果在 `output/arimax_results/`)+ `report/data.md` 初稿
- [ ] **W3 末**:`report/data.md` 最終版 + `setup.md` + reproducibility 重跑通過
