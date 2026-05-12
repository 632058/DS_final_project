---

## 資料前處理：每日國家關係聚合

為了將原始事件級資料（每筆為一則新聞事件）壓縮成適合時序分析的格式，本專案對 `gdelt_events` 進行了以下聚合處理。

### 處理邏輯

原始 GDELT 資料每一列代表一筆新聞事件，同一天、同一對國家、同一種關係類型可能存在數千筆紀錄。我們將其聚合為以下格式：

> **每一列 = 一個日期 × 一對國家 × 一種關係代碼**

欄位對應關係如下：

| 欄位 | 說明 |
|---|---|
| `time_period` | 時間週期（`YYYYMMDD` 整數格式），依聚合粒度不同分別代表當天、當週週一、當月一日 |
| `country_1` / `country_2` | 國家代碼（3 碼），有向圖版本保留 Actor1/Actor2 方向 |
| `relation_code` | 關係代碼，本版本採用 `EventBaseCode`（base 層級） |
| `raw_event_count` | 該週期內該組合的原始事件筆數 |
| `sum_mentions` | 被提及總次數（`NumMentions` 加總） |
| `sum_sources` | 獨立新聞來源總數（`NumSources` 加總） |
| `sum_articles` | 報導文章總數（`NumArticles` 加總） |
| `avg_tone` | 報導平均語氣（`AvgTone` 平均） |
| `avg_goldstein` | 戈德斯坦量表平均分數（`GoldsteinScale` 平均） |

### 重要設計決策

- **時間欄位採用 `DATEADDED` 而非 `SQLDATE`**：避免 Look-ahead bias，確保任何日期的資料只包含到那天為止已被加入資料庫的事件。
- **關係粒度採用 `base`**：`root` 粒度（20 類）過粗，`full`（完整 4 碼）過細且稀疏，`base` 在資訊密度與資料量之間取得較好的平衡。
- **有向圖（directed）**：保留 Actor1→Actor2 方向，反映「誰對誰做什麼」的單向性。
- **同時產出三種時間粒度**：`daily`、`weekly`、`monthly`，可依分析需求切換。


下載連結：
> **daily:**
https://drive.google.com/file/d/1s7IlZA-pM-rEup84vYAfjI0HFqNJpIeS/view?usp=drive_link
> **weekly:**
https://drive.google.com/file/d/1bYMcsUOFnu91IWna7DS6b6EdyU4FUuM7/view?usp=drive_link
> **monthly:**
https://drive.google.com/file/d/1byxQ4iXu2CV67OG0pn2OHUrPfWVkXhJG/view?usp=drive_link

### 讀取檔案

處理後的 Parquet 檔案存放於data資料夾中
讀取方式：

```python
import pandas as pd

df = pd.read_parquet("data/daily_country_relation_directed_base.parquet")
df.head()
```

---

## 衝突事件名單與爆發日標記

本專案整理了一份代表性衝突事件名單，涵蓋 2016 年至 2026 年間共 41 件重要衝突。  
每筆紀錄包含以下欄位：

| 欄位 | 說明 |
|---|---|
| `name` | 衝突事件的識別名稱 |
| `outbreak_date` | 依照公開資料（Wikipedia / Wikidata）核對後的爆發日 |
| `related_countries` | 與此衝突直接相關的國家代碼（GDELT 3 碼格式） |

名單以 JSON 格式儲存於data資料夾中
讀取方式：

```python
import json

with open("data/conflict_events.json", "r", encoding="utf-8") as f:
    events = json.load(f)
```
---

### 如何使用這份名單尋找衝突爆發前的訊號

`outbreak_date` 代表的是衝突被廣泛認定「正式爆發」的那一天。  
真正有價值的問題是：**在那一天之前，資料裡有沒有可以觀察到的徵兆？**

建議的查詢方式是：從 `outbreak_date` 往前取一段時間窗口（例如 30、60、90 天），然後從前處理後的 GDELT 聚合資料中，拉出 `related_countries` 相關國家對在這段時間內的互動記錄。

