# 期末專案提案文件

> **最後更新**：2026-05-06

---

## 1. 專案題目

- **題目**：基於 GDELT 全球事件資料庫之國際事件趨勢分析與預測
- **核心方向**：社會動盪的「前兆捕捉」（Lag / Correlation Analysis）

## 2. 研究問題

### 2.1 修訂版研究問題（建議採用此版本）

> 一國的**媒體報導語氣（AvgTone）**與**事件衝突結構（GoldsteinScale 加權）**升高後，是否能預測該國未來 1–3 個月的抗議事件（CAMEO Root Code = 14）增加？

### 2.2 原版問題

> ~~社會整體負面情緒升高後，是否能預測該地區未來 1–3 個月的抗議事件增加？~~

**修改理由**：見第 5 節「概念釐清」。原版用「社會情緒」描述 AvgTone 與 GoldsteinScale 並不準確，會在 report 答辯時被質疑。

---

## 3. 資料來源

- 資料集：**GDELT 2.0 Event Database**
- 時間範圍：**2018/02 ~ 2026/04**（約 99 個月、約 430 週）
- 儲存格式：**DuckDB**
- 完整版檔案大小：約 70GB
- 主表名：`gdelt_events`

完整資料字典參考既有 `README.md`。本文件僅列出本專案會用到的關鍵欄位。

---

## 4. 關鍵技術原則

### 4.1 預測時一律使用 `DATEADDED`，不可使用 `SQLDATE`

| 欄位 | 意義 | 用途 |
|---|---|---|
| `SQLDATE` | 事件實際發生日 | **僅供描述性分析、不可作為特徵時間切分依據** |
| `DATEADDED` | 事件被加入 GDELT 的時間（YYYYMMDDHHMMSS） | **所有預測模型的特徵與 label 一律以此切分** |

**理由**：新聞會回溯報導過去事件。若以 `SQLDATE` 切分，特徵會「看到未來」，造成 look-ahead bias。本專案需在 Methodology 段落明確聲明使用 `DATEADDED`，反而是加分項。

### 4.2 國家代碼系統

GDELT 使用 **FIPS 10-4 國碼**（不是 ISO 3166）。第一週需先 query 樣本確認，並建好 FIPS↔ISO 對照表。

關鍵國碼速查：

| 國家 | FIPS | 國家 | FIPS |
|---|---|---|---|
| 斯里蘭卡 | CE | 阿根廷 | AR |
| 智利 | CI | 土耳其 | TU |
| 黎巴嫩 | LE | 美國 | US |
| 伊朗 | IR | 委內瑞拉 | VE |

### 4.3 樣本數限制與粒度選擇

- 月度資料：每國 ~99 個月，扣 lag 後 ~90 列 → **XGBoost / Random Forest 會嚴重 overfit**
- **採用對策**：
  1. 主分析改為**週度**（每國 ~430 週）
  2. ML 模型必須做**多國 pooling**（加 country one-hot 變數）
  3. 月度結果僅供 cross-check 使用

---

## 5. 概念釐清（避免常見誤解）

### 5.1 AvgTone ≠ 社會情緒

- AvgTone 是**新聞報導本身的語氣**（journalist / media tone），範圍 −100 到 +100
- 不等同於民眾真實情緒（public sentiment）
- 報告中應明確標示為「**媒體情緒指標 (Media Tone Index)**」

### 5.2 GoldsteinScale 是查表值，不是動態測量

- GoldsteinScale 是 CAMEO 字典中**每個 EventCode 對應的固定值**（−10 到 +10）
- 不是針對單一事件動態評估的
- `AVG(GoldsteinScale)` 實際反映「該期間該國的事件類型組合偏合作還是衝突」
- 報告中應標示為「**衝突結構指標 (Conflict Structure Index)**」而非「情緒」

### 5.3 國內分析 vs 雙邊分析（本專案還需討論是要分析哪一個）

| 分析類型 | 篩選條件 | 對應問題 |
|---|---|---|
| 國內分析 | `WHERE ActionGeo_CountryCode = 'XXX'` | 該國國內情緒 → 該國國內抗議 |
| 雙邊分析 | `WHERE Actor1CountryCode='A' AND Actor2CountryCode='B'`（雙向） | 雙邊緊張 → 雙邊衝突（不在本專案範圍） |

---

## 6. 案例選擇

採用**有明確國內危機的國家**，每國皆走相同 pipeline：

| 國家 | FIPS | 主要案例 | 預期訊號 |
|---|---|---|---|
| 斯里蘭卡 | CE | 2022 經濟崩潰 + 大規模抗議 | 強 |
| 阿根廷 | AR | 長期通膨 + 罷工 | 中強 |
| 智利 | CI | 2019 暴動 | 強 |
| 土耳其 | TU | 經濟危機 + 通膨抗議 | 中 |
| 黎巴嫩 | LE | 2019 起經濟與政治危機 | 強 |

**對照組**（資源允許再加）：日本（JA）、新加坡（SN）、德國（GM）作為穩定型對照。

---

## 7. 分析流程（三層架構）

### 7.1 Layer 1：資料工程（成員 A）

產出三張 DuckDB aggregation table：

#### Table 1：`country_weekly`（主分析用）

```sql
CREATE TABLE country_weekly AS
SELECT
  ActionGeo_CountryCode AS country,
  DATE_TRUNC('week',
    strptime(LEFT(CAST(DATEADDED AS VARCHAR), 8), '%Y%m%d')
  ) AS week_start,
  COUNT(*) AS n_events,
  SUM(NumMentions) AS n_mentions_total,
  AVG(AvgTone) AS avg_tone,
  AVG(GoldsteinScale) AS avg_goldstein,
  -- 加權版本（用 NumMentions 加權，反映重要性）
  SUM(AvgTone * NumMentions) / NULLIF(SUM(NumMentions), 0) AS weighted_tone,
  SUM(GoldsteinScale * NumMentions) / NULLIF(SUM(NumMentions), 0) AS weighted_goldstein,
  -- QuadClass 結構
  SUM(CASE WHEN QuadClass = 1 THEN 1 ELSE 0 END) AS n_verbal_coop,
  SUM(CASE WHEN QuadClass = 2 THEN 1 ELSE 0 END) AS n_material_coop,
  SUM(CASE WHEN QuadClass = 3 THEN 1 ELSE 0 END) AS n_verbal_conf,
  SUM(CASE WHEN QuadClass = 4 THEN 1 ELSE 0 END) AS n_material_conf,
  -- 核心標的
  SUM(CASE WHEN EventRootCode = '14' THEN 1 ELSE 0 END) AS protest_count,
  -- 升級訊號
  SUM(CASE WHEN EventRootCode IN ('18','19','20') THEN 1 ELSE 0 END) AS violence_count,
  -- 言語前兆訊號
  SUM(CASE WHEN EventRootCode IN ('10','11') THEN 1 ELSE 0 END) AS verbal_threat_count
FROM gdelt_events
WHERE ActionGeo_CountryCode IN ('CE','AR','CI','TU','LE')
GROUP BY country, week_start
ORDER BY country, week_start;
```

#### Table 2：`country_monthly`（次分析、cross-check 用）

結構同 Table 1，但 `DATE_TRUNC('month', ...)`。

#### Table 3：`country_eventmix_weekly`（降維視覺化用）

每國每週 20 維 root code 比例向量 + Tone/Goldstein 統計量。

**重要**：以上三張表**所有時間欄位皆以 `DATEADDED` 為基礎**，不使用 `SQLDATE`。

### 7.2 Layer 2：特徵工程（成員 B）

對 `country_weekly` 新增以下欄位，產出 `feature_matrix`：

#### 滾動統計量
- `tone_rolling_mean_4w`, `tone_rolling_mean_8w`, `tone_rolling_mean_12w`
- `tone_rolling_std_4w`, `tone_rolling_std_8w`
- 同樣對 `avg_goldstein`、`n_material_conf` 計算

#### 變化率
- `tone_diff_1w`（與上週差距）
- `tone_diff_4w`（與 4 週前差距）
- `tone_pct_change_4w`

#### Lag 變數
- `tone_lag1` ~ `tone_lag12`（週度，1–12 週）
- `goldstein_lag1` ~ `goldstein_lag12`
- `material_conf_lag1` ~ `material_conf_lag12`

#### 缺值處理（重要：依欄位區分）

| 欄位類型 | 缺值處理 |
|---|---|
| `protest_count`, `violence_count` 等 count 類 | 補 0（無事件 = 0 件） |
| `avg_tone`, `avg_goldstein` 等平均值類 | **不可補 0**（0 是中性值，會污染訊號）→ 使用前向填補 (`ffill`) 或國家整體均值 |
| `weighted_*` | 同上，前向填補 |

#### 時間連續性
確保每國從 2018-02 第一週到最新週皆有列，缺失週按上述規則補齊。

### 7.3 Layer 3：分析與建模

#### 7.3.1 成員 C：時間序列建模

1. **平穩性檢定**：對每國 `protest_count`、`avg_tone`、`avg_goldstein` 跑 ADF test
2. **差分處理**：非平穩序列做一階差分，必要時季節性差分
3. **ARIMAX 建模**：
   - 標的：`protest_count`
   - 外生變數 (exog)：`avg_tone`, `avg_goldstein`, `n_material_conf`（含 lag）
   - 模型階數選擇：以 AIC / BIC 選 (p, d, q)
4. **模型診斷**：殘差 Ljung-Box 白噪檢定、ACF / PACF plot

#### 7.3.2 成員 D：統計檢定（與成員 C 平行）

1. **Cross-Correlation Function (CCF)**：
   - 對每國計算 `avg_tone` 與 `protest_count` 在 lag 0–12 週的相關係數
   - 找出最佳 lag
2. **Granger Causality Test**：
   - H₀：avg_tone does not Granger-cause protest_count
   - 對每國跑、報告 F-test 與 p-value
3. **產出 Fig 4：CCF heatmap**（x=lag, y=country, color=correlation）

#### 7.3.3 成員 E：機器學習建模

1. **任務性質**：迴歸（預測下週 `protest_count`）
2. **模型選擇優先序**：
   1. **Lasso / Ridge**（baseline，必跑）
   2. **XGBoost Regressor**（主模型）
3. **Pooling 策略**：5 國資料合併訓練，加 country one-hot 特徵（**必做**，否則樣本數不足）
4. **時間切分**：以 `DATEADDED` 切，前 80% train、後 20% test，**不可隨機切**
5. **評估指標**：RMSE、MAE、Directional Accuracy（漲跌方向命中率）
6. **可選**：SHAP 解釋特徵重要性

#### 7.3.4 成員 F：視覺化

##### 必做圖表（迴歸任務適用）

| Fig | 內容 |
|---|---|
| Fig 1 | 雙軸折線圖（每國一張，左軸 tone、右軸 protest_count） |
| Fig 2 | Goldstein / Tone 時序 heatmap（行=國、列=週、色=值） |
| Fig 3 | 國家事件混合 UMAP / t-SNE（每點 = 「某國某週」） |
| Fig 4 | CCF heatmap（成員 D 結果） |
| Fig 5 | 預測 vs 實際散點圖 + 對角線 |
| Fig 6 | Residual plot（檢查模型偏誤） |
| Fig 7 | 時間序列 forecast 圖（藍=實際、紅=預測 + 95% CI） |
| Fig 8 | SHAP summary plot |

##### ⚠ 不建議使用的視覺化

- **ROC 曲線**（ROC 是分類任務的圖、本專案是迴歸）
- **Confusion Matrix**（同上）
- 除非任務改成「預測下個月是否爆發抗議潮」二元分類，才使用 ROC (見第10節)

##### 可選
- 地圖視覺化（Folium / GeoPandas）：標示各國抗議事件熱度

---

## 8. 分工總表

| 成員 | 負責工作 |
|---|---|
| A | 確認 FIPS 國碼；建立三張聚合表（`country_weekly` 優先交、`country_monthly`、`country_eventmix_weekly`）；撰寫 schema doc；報告 Data 章節 |
| B | Week 1 依 schema 先寫 feature engineering 程式框架；收到 `country_weekly` 後執行，交出 `feature_matrix`；Week 3 起支援 C/D/E 的 feature 問題；報告 Feature Engineering 章節 |
| C | Week 1 準備 ARIMAX 程式框架；ADF 平穩性檢定（含差分處理）；ARIMAX 建模 + 模型診斷（Fig 8）；報告 Methodology / Time Series 章節 |
| D | Week 1 準備 CCF + Granger 程式框架；CCF（lag 0–12 週）+ Granger Causality Test（Fig 5）；報告統計結果章節 |
| E | Week 1 準備 Lasso/XGBoost 程式框架；Lasso/Ridge baseline + XGBoost（Fig 6, 7）；SHAP 分析（Fig 9）；報告 ML 章節 |
| F | Week 1 建 repo 結構、視覺化模板；EDA 圖（Fig 1, 2）；降維視覺化（Fig 3）；Week 4 整合所有圖；Week 3 起撰寫報告框架 |

---

## 9. 5 週時程（壓縮並行版）

> **設計原則**：A 優先交 `country_weekly`，讓 B、F 立刻開工；B 在等真實資料時先寫程式框架；C、D、E 從 Week 3 起完全並行；F 每週皆有產出，不等人。

| 週次 | 成員 A | 成員 B | 成員 C | 成員 D | 成員 E | 成員 F | 週末交付 |
|---|---|---|---|---|---|---|---|
| **1** | 確認 FIPS 國碼、建 `country_weekly` | 依 schema 寫 feature engineering 程式框架 | 準備 ARIMAX 程式框架 | 準備 CCF + Granger 程式框架 | 準備 Lasso / XGBoost 程式框架 | 建 repo 結構、視覺化模板 | `country_weekly`（A）、各成員程式框架 |
| **2** | 完成 `country_monthly` + schema doc | 收到 `country_weekly` 後立刻執行，交出 `feature_matrix` | ADF test（先跑 `country_weekly` 原始版） | ADF test（先跑 `country_weekly` 原始版） | 準備 train/test 切分邏輯 | Fig 1, Fig 2 | `feature_matrix`（B）、Fig 1,2（F）、schema doc（A） |
| **3** | 建 `country_eventmix_weekly`；**協助 C 跑 ARIMAX**（負責 2–3 國） | 支援 C/D/E 的 feature 問題 | ARIMAX 框架 + 跑 2–3 國 + 整合 Fig 7 | CCF + Granger（Fig 4）；**協助 E 跑 Lasso baseline** | XGBoost 主模型（Fig 5, 6） | Fig 3（UMAP / t-SNE）、撰寫報告框架 | Fig 3,4,5,6,7、統計檢定報表 |
| **4** | 撰寫 Data 章節 | 撰寫 Feature Engineering 章節 | 完成模型、撰寫 Methodology 章節 | **協助 E 跑 SHAP**；撰寫統計結果章節 | 撰寫 ML 章節 | 整合所有圖 | 完整 figure set（Fig 8 SHAP）、各章節草稿 |
| **5** | **全員**：Report 最終整合、簡報製作、reproducibility check | | | | | | 最終交付 |

---

## 10. 待確認事項（送組員會議決定）

- [ ] 分析方向：國內情勢分析（`ActionGeo_CountryCode`）還是國家間情勢分析（`Actor1CountryCode` ↔ `Actor2CountryCode`）？（見第 5.3 節）
- [ ] 預測任務是否同時做迴歸（預測 `protest_count` 數量）與分類（預測「下個月是否爆發抗議潮」）兩個版本？迴歸版本已規劃完整，分類版本可加入 ROC / AUC 評估
- [ ] Repo 是否使用 GitHub Actions 跑 reproducibility check

---

## 附錄 A：本專案使用的 GDELT 欄位

| 欄位 | 型別 | 用途 |
|---|---|---|
| `GLOBALEVENTID` | int64 | 主鍵 |
| `SQLDATE` | int64 | 事件發生日（**僅描述性、不用於切分**） |
| `DATEADDED` | int64 | 入庫時間（**所有預測切分用此**） |
| `ActionGeo_CountryCode` | varchar | 事件實際發生地 FIPS 國碼（**國內分析用此欄**） |
| `Actor1CountryCode` | varchar | 行為者 1 國碼 |
| `Actor2CountryCode` | varchar | 行為者 2 國碼 |
| `EventCode` | varchar | CAMEO 細分代碼 |
| `EventRootCode` | varchar | CAMEO 根代碼（'14' = Protest） |
| `QuadClass` | int64 | 1=言語合作 / 2=實質合作 / 3=言語衝突 / 4=實質衝突 |
| `GoldsteinScale` | double | −10 至 +10，CAMEO 字典查表值 |
| `AvgTone` | double | −100 至 +100，媒體報導語氣 |
| `NumMentions` | int64 | 提及次數（重要性權重用） |
| `NumSources` | int64 | 獨立來源數 |
| `NumArticles` | int64 | 報導文章數 |

## 附錄 B：CAMEO 重要 Root Code

| Root Code | 類別 | 本專案用途 |
|---|---|---|
| 01 | MAKE PUBLIC STATEMENT | 言語訊號 |
| 02 | APPEAL | 言語合作 |
| 10 | DEMAND | 言語衝突（leading indicator 候選） |
| 11 | DISAPPROVE | 言語衝突（leading indicator 候選） |
| **14** | **PROTEST** | **核心 Y 變數** |
| 17 | COERCE | 強制行為 |
| 18 | ASSAULT | 升級訊號 |
| 19 | FIGHT | 升級訊號 |
| 20 | USE UNCONVENTIONAL MASS VIOLENCE | 戰爭訊號 |

完整代碼：`http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf`

---

**文件結束**。後續修改請直接編輯本文件並更新版本號。
