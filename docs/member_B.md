# 成員 B 工作手冊

> 你負責**特徵工程**:把 A 的 `country_weekly` 變成 ML / 時序模型可用的 `feature_matrix`,加上 rolling、lag、變化率、country one-hot,並做缺值處理。

---

## 全部交付物總覽

| 週次 | 交付物 | 對誰交付 | 截止 |
|---|---|---|---|
| W1 Day 1-3 | 熟悉 `src/feature_engineering.py` + 跑通 `tests/test_feature_engineering.py` | 自己 | Day 3 結束 |
| W1 Day 4-7 | `output/feature_matrix.parquet` v1 | E、C、D | Day 7 結束 |
| W2 中 | `feature_matrix` v2(回應 E 的補強建議) | C、D、E | Week 2 中 |
| W2 末 | `report/feature_engineering.md` 初稿 + 支援 C/D/E 的 feature 問題 | F | Week 2 末 |
| W3 末 | 章節最終版 + 協助整合 | 全組 | Week 3 末 |

---

## 依賴關係

- **你依賴 A**:Day 4 起需要 `output/country_weekly.parquet`。
- **依賴你的人**:E(baseline 必須用 `feature_matrix` v1)、C、D(W2 起用 v2)。
- **Day 1-3 不等 A**:你可以先讀程式、跑測試。

---

## Week 1 Day 1-3:熟悉框架 + 跑通測試

### 你會用到的檔案
- `src/feature_engineering.py`(已備好)
- `tests/test_feature_engineering.py`(已備好)
- `scripts/04_build_feature_matrix.py`(Day 4 開始用)

### 步驟

#### Step 1:讀懂 `src/feature_engineering.py`
- 主要 function:`build_feature_matrix(df_weekly)`
- 內部分 6 步:`ensure_continuous_weeks` → `impute_missing` → `add_rolling_features` → `add_lag_features` → `add_diff_features` → `add_country_one_hot`
- 每個 helper 都寫了 docstring,讀一遍

#### Step 2:跑測試確認程式可運作
- 執行 `pytest tests/test_feature_engineering.py -v`
- 應該全綠,確認 dummy 資料下:
  - `ensure_continuous_weeks` 補齊缺週
  - `impute_missing` count 補 0、average 用 ffill
  - `add_rolling_features` 用 `shift(1)` 不洩漏未來
  - `add_lag_features` 第 N 週的 lag1 = 第 N-1 週的值
  - `add_country_one_hot` 不刪 country 欄位

#### Step 3:讀懂滾動窗口設計
- 預設窗口:`ROLLING_WINDOWS = [4, 8, 12]`(在 `src/constants.py`)
- 預設 lag:`LAG_RANGE = list(range(1, 13))`(1~12 週)
- 想改參數就改 `src/constants.py`,不要 hardcode 在 script 裡

---

## Week 1 Day 4-7:產出 `feature_matrix` v1

### 步驟

#### Step 1:確認 A 的資料已到位
- 看 `output/country_weekly.parquet` 是否存在
- 跑 `python scripts/sanity_check.py` 應顯示 [PASS]

#### Step 2:執行 feature engineering
- 執行 `python scripts/04_build_feature_matrix.py`
- 預期產出 `output/feature_matrix.parquet`
- 應印出 shape(列數約 5 × 430 = 2150,欄位數約 80+)

#### Step 3:Sanity check
- 確認 `feature_matrix` 沒有 inf(已在 `sanity_check.py`)
- 看 `df.head(20)` 確認 lag1 / rolling 第一週為 NaN(預期)
- 看 `df['avg_tone_lag1'].isna().sum()`,應該等於國家數(每國第一週 NaN)

#### Step 4:通知 E 開工
- 群組通知:
  > @E `feature_matrix.parquet` v1 已產出,你可以跑 `scripts/09_train_lasso.py` 了。

---

## Week 2:v2 補強 + 章節 + 跨成員支援

### Task 1:回應 E 的補強建議

E 在 W1 末跑完 baseline 後可能會說:「v1 RMSE = X,建議加 [feature]」。常見補強:
- **互動項**:`tone × goldstein`(媒體語氣 × 衝突結構)
- **比例**:`n_material_conf / n_events`(衝突事件佔比)
- **時間特徵**:`week_of_year`、`month`(季節性)
- **更長 lag**:若 D 的 CCF 顯示 lag > 12 有訊號,加 lag13-24

修改方式:擴充 `src/feature_engineering.py` 加新 helper,然後重跑 `scripts/04_build_feature_matrix.py`。

### Task 2:支援 C/D/E 的 feature 問題

| 成員 | 可能會問你 |
|---|---|
| C(ARIMAX) | 「我要某國的 protest_count + 3 個 exog 的 lag,怎麼從 feature_matrix 抽?」 |
| D(Granger) | 「我要乾淨的 avg_tone 與 protest_count 兩條時序,按國家分開」 |
| E(XGBoost) | 「country one-hot 缺了某國 dummy column?」(預設 drop_first=False,應該每國都有) |

可加 helper 到 `src/data_loader.py`:
```python
def get_country_series(country: str, cols: list) -> pd.DataFrame: ...
```

### Task 3:撰寫 `report/feature_engineering.md` 初稿

骨架已在 `report/feature_engineering.md`,填內容:
1. **特徵設計動機**:為什麼用 rolling、lag、diff(前兆訊號需要時間結構)
2. **缺值處理**:為何 count 補 0、average 用 ffill
3. **時間連續性**:每國從第一週到最新週皆有列
4. **避免 look-ahead bias**:所有 rolling 都 `shift(1)`
5. **Pooling 策略**:country one-hot 的目的(樣本不足要 pooling)
6. **特徵清單**:用表格列出所有產出特徵

---

## Week 3:章節最終版 + 協助整合

- 跟 E 對齊:ML 章節對 feature 的描述不要跟你的章節衝突
- 跟 F 對齊:報告整體章節順序與標題層級
- 補 reproducibility 段:「執行 `python scripts/04_build_feature_matrix.py` 即可重現」

---

## 常見陷阱

| 陷阱 | 後果 | 解法 |
|---|---|---|
| Rolling 沒 `shift(1)` | Look-ahead,特徵看到當週的值 | `src/feature_engineering.py` 已內建 `s.shift(1).rolling(w).mean()` |
| 跨國 leak(rolling 跨 country) | 用其他國的資料污染本國 | 一律 `df.groupby('country')[col].transform(...)` |
| Average 欄位補 0 | 0 是中性值,污染訊號 | `impute_missing` 已分流:count 補 0、average 用 ffill |
| 缺週直接 drop | C 的 ARIMAX 需要連續時間 | `ensure_continuous_weeks` 已補齊 |
| One-hot 把 `country` 欄位刪掉 | C/D 仍需要原始 country 欄位 | 已用 `df.join(one_hot)` 保留 |
| `pct_change` 出現 inf | 除以 0 | 已 `.replace([np.inf, -np.inf], np.nan)` |
| 週起日跟 A 的 DuckDB `DATE_TRUNC` 不一致 | merge 不到 | 跟 A 確認週起日(預設 W-MON);若不對改 `freq` 參數 |

---

## 交付檢查清單

- [ ] **W1 Day 3**:讀完 `src/feature_engineering.py` + `pytest tests/test_feature_engineering.py` 全綠
- [ ] **W1 Day 7**:`output/feature_matrix.parquet` v1 + sanity check pass + 通知 E
- [ ] **W2 中**:回應 E 補強建議,產出 v2
- [ ] **W2 末**:`report/feature_engineering.md` 初稿 + 已回應 C/D/E 的 feature 問題
- [ ] **W3 末**:章節最終版
