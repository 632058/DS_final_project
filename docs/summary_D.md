
---

## 1. 研究問題

> 媒體報導語氣（AvgTone）與衝突結構指標（GoldsteinScale）惡化後，是否能預測該國國內抗議事件（CAMEO Root Code = 14）在未來數週增加？

分析分為兩個主軸：
- **主軸 A**：五個目標國的國內抗議前兆分析
- **主軸 B**：國際衝突升級前兆分析（CAMEO code > 15）

---

## 2. 資料來源與處理

### 2.1 來源檔案
- `weekly_country_relation_directed_base.parquet`
- `conflict_events.json`



### 2.2 資料決策

**決策 1：protest_count 限定國內（country_1 == country_2）**  
原因：跨國抗議（如土耳其人抗議以色列）占土耳其所有抗議的 74%，與國內社會動盪無關。  
robustness check 欄位 `protest_count_all` 保留供比較。

**決策 2：avg_tone / avg_goldstein 不限國內**  
原因：媒體語氣應反映整體報導環境，不只是國內互動。


**國家代碼對應**：

| ISO 3碼 | FIPS | 國家 |
|---|---|---|
| ARG | AR | 阿根廷 |
| CHL | CI | 智利 |
| TUR | TU | 土耳其 |
| LBN | LE | 黎巴嫩 |
| LKA | CE | 斯里蘭卡 |

---

## 3. 分析方法

### 3.1 CCF（Cross-Correlation Function）

**輸出**：`output/ccf_granger/ccf_{x_var}_to_protest.parquet`、`figures/fig4_ccf_heatmap.png`

**計算公式**：
```
CCF(k) = corr( x(t-k), protest(t) )   k = 0, 1, ..., 12 週
```
lag_k 的正值代表：x 在 k 週前的值與當週抗議數的相關係數。

**窗口設計（重要決策）**：

| 版本 | 窗口 | 問題 |
|---|---|---|
| 現行| **爆發前 52 週 + 爆發後 12 週** | X（tone 惡化）在前，Y（protest 爆發）在後，符合因果方向 |

**窗口結構示意**：
```
──────────────────────────────┬──────────────
  tone 逐漸惡化（X 訊號）      │ 爆發日  protest 爆發（Y）
──────────────────────────────┴──────────────
        爆發前 52 週                爆發後 12 週
```


### 3.2 Granger Causality Test

**輸出**：`output/ccf_granger/granger_summary.parquet`

H₀：x 的歷史值不能改善對 protest_count 的預測（x 不 Granger-cause protest）  
H₁：拒絕 H₀，x 具有統計上的因果可預測性

**前處理**：以 ADF 檢定自動差分至平穩（`stationarity_diff()`），回報差分階數 d。

**窗口設計**：與 CCF 一致，爆發前 52 週 + 爆發後 12 週（共 64 週），確保兩個方法可直接對比。  
max_lag = 12 週，報告各 lag 中最小 p-value（SSR-based F-test）。

### 3.3 爆發日設定

| 國家 | 爆發日 | 事件 | 選定依據 |
|---|---|---|---|
| 斯里蘭卡 (CE) | 2022-03-31 | 2022 年經濟崩潰大規模抗議 | 週 protest 數從 2 急升至 71 |
| **阿根廷 (AR)** | **2018-08-06** | 比索危機（IMF 緊急求援）| 全序列最高週（15 件） |
| 智利 (CI) | 2019-10-18 | estallido social（18-O）| 次週衝至 102 件 |
| **土耳其 (TU)** | **2021-02-01** | Bogaziçi 大學抗議 | 全序列最高週（83 件），非通膨抗議 |
| 黎巴嫩 (LE) | 2019-10-17 | 十月革命（17-O）| 次週衝至 166 件 |


---

## 4. 主軸 A：國內抗議分析結果

### 4.1 CCF 結果（avg_tone → protest_count）

| 國家 | lag_0 | lag_1 | 最大 |r| | 最佳 lag | 方向 |
|---|---|---|---|---|---|
| **斯里蘭卡** | **−0.60** | −0.29 | 0.60 | lag_0 |  負相關 |
| **智利** | **−0.58** | −0.51 | 0.58 | lag_0 |  負相關 |
| **黎巴嫩** | −0.45 | −0.38 | **0.55** | lag_9 |  負相關|
| 土耳其 | +0.09 | +0.12 | 0.30 | lag_4 |  正相關|
| 阿根廷 | −0.21 | +0.10 | 0.21 | lag_0 | ➖ 弱 |

**黎巴嫩**：lag_0 到 lag_9 全程維持負相關，且在 lag_8（−0.53）、lag_9（−0.55）最強，代表語氣惡化 **8–9 週前**就能預測抗議爆發。

**土耳其正相關解讀**：Bogaziçi 抗議為政治任命觸發型，抗議發生前媒體語氣並未持續惡化，屬於政治事件型而非社會積累型，不符合本研究假說框架。

### 4.2 CCF 結果（avg_goldstein / n_material_conf）

| 國家 | avg_goldstein 最強 | n_material_conf 最強 |
|---|---|---|
| 斯里蘭卡 | lag_0 = −0.37 | lag_0 = **+0.72** |
| 阿根廷 | lag_0 = −0.21 | 弱 |
| 智利 | lag_0 = **−0.52** | lag_0 = **+0.72** |
| 土耳其 | lag_12 = +0.25（反向）| 弱 |
| 黎巴嫩 | lag_8 = **−0.57** | lag_8 = **+0.61** |

黎巴嫩三個指標在 lag_8–9 周圍全部顯著，跨指標一致性極高。

### 4.3 Granger 因果檢定結果（64 週窗口）

| 國家 | avg_tone | avg_goldstein | n_material_conf |
|---|---|---|---|
| **斯里蘭卡** |  lag=1, p=0.004 |  p=0.381 |  p=0.475 |
| **智利** |  lag=4, p=0.023 |  lag=7, p=0.016 | p=0.075 |
| **黎巴嫩** |  lag=9, p=0.003 |  lag=10, p=0.003 |  lag=10, **p=2e-7** |
| 土耳其 | p=0.084 |  p=0.167 |  p=0.670 |
| 阿根廷 |  p=0.359 |  p=0.226 |  p=0.600 |

**重要**：黎巴嫩 avg_tone 的 Granger best_lag = 9，與 CCF 最大 |r| 的 lag_9 完全一致，兩個獨立方法指向同一窗口。


---

## 5. 主軸 B：國際衝突升級分析

**輸出**：`output/ccf_granger/conflict_{ccf|granger}_results.parquet`、`figures/fig_conflict_ccf_heatmap.png`

**Y 變數**：CAMEO root code > 15（COERCE=17 / ASSAULT=18 / FIGHT=19 / MASS VIOLENCE=20）  
**窗口**：與主軸 A 一致，**爆發前 52 週 + 爆發後 12 週（共 64 週）**  


### 5.1 選取的 5 個衝突

| 衝突 | 國家對 | 爆發日 |
|---|---|---|
| 俄羅斯入侵烏克蘭 | RUS↔UKR | 2022-02-24 |
| 加薩戰爭 2023 | ISR↔PSE | 2023-10-07 |
| 第二次納卡戰爭 | ARM↔AZE | 2020-09-27 |
| 印巴軍事對峙 | IND↔PAK | 2019-02-14 |
| 美伊危機（蘇萊曼尼）| USA↔IRN | 2020-01-03 |

### 5.2 CCF 結果（avg_tone，64 週窗口）

| 衝突 | 最佳 lag | r | 解讀 |
|---|---|---|---|
| ARM↔AZE | lag_0 | **−0.578** | 強烈同步負相關 |
| IND↔PAK | lag_1 | **−0.466** | 語氣提前 1 週惡化 |
| USA↔IRN | lag_1 | **−0.432** | 語氣提前 1 週惡化 |
| RUS↔UKR | lag_0 | −0.305 | 中等（地緣政治決策型）|
| ISR↔PSE | lag_3 | +0.290 | 弱正相關（爆發後期窗口效應）|

**ISR↔PSE 正相關說明**：加薩戰爭爆發後 12 週窗口內，衝突強度持續攀升，此時媒體語氣亦受戰事波動，lag_3 正相關反映的是同步升溫而非前兆。

### 5.3 Granger 因果檢定結果（64 週窗口）

| 衝突 | avg_tone | avg_goldstein | n_material_conf* |
|---|---|---|---|
| RUS↔UKR | ✗ p=0.507 | ✗ p=0.313 |  lag=12, p=0.0002 |
| ISR↔PSE | ✗ p=0.228 | ✗ p=0.479 | ✗ p=0.144 |
| **ARM↔AZE** | ** lag=2, p=0.012** | ✗ p=0.159 |  lag=11, p=0.0002 |
| **IND↔PAK** | ** lag=3, p=0.010** | ✗ p=0.607 | lag=3, p<0.0001 |
| **USA↔IRN** | ** lag=12, p=0.020** | ** lag=2, p=0.037** | lag=2, p<0.0001 |

\* n_material_conf 與 Y 高度重疊，顯著結果不代表獨立前兆

**顯著比例（排除 n_material_conf）**：avg_tone 3/5 顯著、avg_goldstein 1/5 顯著







