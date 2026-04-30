# GDELT 新聞事件資料庫（DuckDB）

這份資料集包含 GDELT（Global Database of Events, Language, and Tone）的新聞事件紀錄，並已轉換為高效能的 DuckDB 格式，方便直接使用 Python 進行資料分析與查詢

## 檔案下載

為了方便實驗與分析，這裡提供兩個版本的資料庫檔案：

1. **輕量實驗版（建議先使用）**  
   已經過篩選的資料集，檔案較小，適合測試程式碼與熟悉欄位 
   - https://drive.google.com/file/d/1rT7ho_RXkcK8J4RYyKBqP4fh14FrAUqg/view?usp=drive_link

2. **完整版資料庫**  
   包含全量資料，檔案較大，適合final project  
   - https://drive.google.com/file/d/1emP2Ezg7ddljCogeqIDrYajucPEPfPiv/view?usp=drive_link

---

## 環境安裝

使用 `duckdb` 讀取資料。

請在Terminal執行以下指令：

```bash
pip install duckdb pandas
```

## Quick Start

下載好資料庫檔案（例如 `gdelt_filtered_20251001_20260428.duckdb`）並放在與程式碼同一個資料夾後，可以執行以下 Python 程式碼查看前 20 筆資料與一些屬性：

```python
import duckdb

db_path = "gdelt_filtered_20251001_20260428.duckdb"
con = duckdb.connect(db_path)

# 1. 查詢總筆數、最小時間 (最早) 與最大時間 (最新)
stats = con.sql("""
    SELECT 
        COUNT(*) AS total_rows,
        MIN(DATEADDED) AS min_date,
        MAX(DATEADDED) AS max_date
    FROM gdelt_events
""").fetchone()

total_rows = stats[0]
min_date = stats[1]
max_date = stats[2]

# 印出摘要資訊
print("=== 資料庫摘要資訊 ===")
print(f"總資料筆數: {total_rows:,} 筆")
print(f"最早 DATEADDED: {min_date}")
print(f"最新 DATEADDED: {max_date}")
print("======================\n")

# 2. 查詢 gdelt_events 的前 20 筆資料
print("查詢 gdelt_events 的前 20 筆資料：\n")
con.sql("SELECT * FROM gdelt_events LIMIT 20").show()

con.close()
```


### 應用範例：繪製每日事件趨勢圖 (Matplotlib)

如果你想進一步視覺化分析資料，例如「計算美國和伊朗之間發生 `EventCode > 15` 的事件總數並畫出折線圖」(EventCode是一種CAMEO代碼，詳見後面說明)，你可以結合 `pandas` 與 `matplotlib` 來完成

請先確保你已經安裝了繪圖套件：
`pip install matplotlib`

接著執行以下程式碼：

```python
import duckdb
import pandas as pd
import matplotlib.pyplot as plt

# 指定資料庫檔案路徑
db_path = "gdelt_filtered_20251001_20260428.duckdb"
con = duckdb.connect(db_path)

print("正在彙整美國與伊朗之間的互動資料並繪製圖表...\n")

# SQL 查詢邏輯修正：
# 加入國家過濾條件，確保 Actor1 與 Actor2 分別為 USA 與 IRN (雙向皆包含)
query = """
    SELECT 
        SUBSTRING(CAST(DATEADDED AS VARCHAR), 1, 8) AS event_date,
        COUNT(*) AS daily_count
    FROM gdelt_events 
    WHERE TRY_CAST(EventCode AS INTEGER) > 15
      AND (
          (Actor1CountryCode = 'USA' AND Actor2CountryCode = 'IRN') OR
          (Actor1CountryCode = 'IRN' AND Actor2CountryCode = 'USA')
      )
    GROUP BY event_date
    ORDER BY event_date
"""

# 執行查詢並將結果轉為 DataFrame
df = con.sql(query).df()
con.close()

# 檢查是否有資料
if df.empty:
    print("在此篩選條件下找不到任何資料，請檢查 EventCode 或國家代碼。")
else:
    # 轉換日期格式
    df['event_date'] = pd.to_datetime(df['event_date'], format='%Y%m%d')

    # --- 繪製折線圖 ---
    plt.figure(figsize=(12, 6))
    plt.plot(df['event_date'], df['daily_count'], marker='o', linestyle='-', color='#e74c3c', linewidth=2)

    # 設定標題與標籤
    plt.title('Daily Trend of USA-Iran Interactions (EventCode > 15)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Events', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()
```

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/9535f794-38e6-40f3-b48e-4cda68163296" />



### 注意事項

duckdb雖然是一個檔案但操作方法近似sql，若不太熟悉的話，建議可以把前面的基礎查詢code、最下方的「資料字典」連同你的需求直接貼給AI，請他來撰寫會最精準

請注意“SQLDATE”欄位是事件發生日期，“DATEADDED”欄位才是被加入資料庫的日期，也就是說有時候新聞會回顧過去的事件導致最新的“DATEADDED”但“SQLDATE”是過去的時間，所以建議要做“預測“的話，一律以“DATEADDED”為準否則會有”Look ahead bias“

## CAMEO 編碼 (Conflict and Mediation Event Observations) 說明

CAMEO（Conflict and Mediation Event Observations，衝突與調停事件觀察）是一種用於對**國際政治和社會事件進行自動化分類與編碼**的框架系統。當自然語言處理系統在分析新聞報導時（例如讀到：「A國對B國實施經濟制裁」或「C群眾在廣場抗議」），就會運用 CAMEO 字典將這些非結構化的文本轉換為標準化的數據代碼

用CAMEO代碼我們可以直接知道該事件大概發生了什麼事情不用細看，當然本次專案中不一定要用CAMEO來做，有很多其他欄位可以做分析，但CAMEO是最常用的

### 事件編碼 (Event Codes)
CAMEO 採用**階層式（樹狀）架構**來定義「誰對誰做了什麼動作」。事件代碼通常由 2 到 4 位數字組成，層級越深，描述越具體

最頂層有 20 個**根代碼 (Root Codes)**，涵蓋了人類社會中主要的互動行為：
* `01` 到 `05`：**口頭合作**（如：發表聲明、呼籲、拜訪）
* `06` 到 `09`：**實質合作**（如：提供援助、交還領土）
* `10` 到 `14`：**口頭衝突**（如：要求、威脅、抗議）
* `15` 到 `20`：**實質衝突**（如：展示軍力、實施制裁、使用武力）

**階層範例**：
* `14`：抗議 (Protest)
  * `141`：示威或集會 (Demonstrate or rally)
    * `1411`：為了領導層換屆而示威 (Demonstrate for leadership change)

### 官方說明文件與參考連結

如果你需要深入了解每個代碼的具體定義，或者在分析資料時需要對照字典，請參考以下官方文件：

### 核心編碼手冊
* **CAMEO Conflict and Mediation Event Observations Manual** (PDF)
  官方說明書，包含所有代碼的定義、編碼規則以及判斷標準
  [http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf](http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf)


## 資料字典（欄位說明）

以下是 `gdelt_events` 資料表的主要欄位、資料型態與意義說明：

| 欄位名稱（Column） | 資料型態 | 欄位說明（Description） |
|---|---|---|
| `GLOBALEVENTID` | `int64` | 全球事件的唯一識別碼 |
| `SQLDATE` | `int64` | 事件發生日期（格式：`YYYYMMDD`） |
| `MonthYear` | `int64` | 事件發生年月（格式：`YYYYMM`） |
| `Year` | `int64` | 事件發生年份（格式：`YYYY`） |
| `FractionDate` | `double` | 事件發生時間的年份小數表示法 |
| `Actor1Code` | `varchar` | 參與者 1（發起方）的完整身分代碼 |
| `Actor1Name` | `varchar` | 參與者 1 的實際名稱（如國家名、人名、組織名） |
| `Actor1CountryCode` | `varchar` | 參與者 1 的國家代碼（3 碼） |
| `Actor1KnownGroupCode` | `varchar` | 參與者 1 的已知國際組織代碼（如 `UN`、`EU`） |
| `Actor1EthnicCode` | `varchar` | 參與者 1 的族群代碼 |
| `Actor1Religion1Code` | `varchar` | 參與者 1 的主要宗教代碼 |
| `Actor1Religion2Code` | `varchar` | 參與者 1 的次要宗教代碼 |
| `Actor1Type1Code` ～ `Actor1Type3Code` | `varchar` | 參與者 1 的類型代碼（如政府、媒體、警察等，最多 3 個） |
| `Actor2Code` | `varchar` | 參與者 2（接收方）的完整身分代碼 |
| `Actor2Name` | `varchar` | 參與者 2 的實際名稱 |
| `Actor2CountryCode` | `varchar` | 參與者 2 的國家代碼（3 碼） |
| `Actor2KnownGroupCode` | `varchar` | 參與者 2 的已知國際組織代碼 |
| `Actor2EthnicCode` | `varchar` | 參與者 2 的族群代碼 |
| `Actor2Religion1Code` | `varchar` | 參與者 2 的主要宗教代碼 |
| `Actor2Religion2Code` | `varchar` | 參與者 2 的次要宗教代碼 |
| `Actor2Type1Code` ～ `Actor2Type3Code` | `varchar` | 參與者 2 的類型代碼（最多 3 個） |
| `IsRootEvent` | `int64` | 標記此事件是否為根事件（`1` 代表是，`0` 代表否） |
| `EventCode` | `varchar` | 具體的 CAMEO 事件分類代碼 |
| `EventBaseCode` | `varchar` | 基礎 CAMEO 事件代碼 |
| `EventRootCode` | `varchar` | 根 CAMEO 事件代碼（最高層級分類） |
| `QuadClass` | `int64` | 事件四大類：`1` 口頭合作、`2` 實質合作、`3` 口頭衝突、`4` 實質衝突 |
| `GoldsteinScale` | `double` | 戈德斯坦量表分數（`-10` 到 `+10`，衡量事件的衝突或合作程度） |
| `NumMentions` | `int64` | 此事件在新聞網路中被提及的總次數 |
| `NumSources` | `int64` | 報導此事件的獨立新聞來源數量 |
| `NumArticles` | `int64` | 報導此事件的新聞文章總數 |
| `AvgTone` | `double` | 報導此事件的平均語氣／情緒分數（負值代表負面，正值代表正面） |
| `Actor1Geo_Type` | `int64` | 參與者 1 的地理層級（`1`：國家、`2`：州／省、`3`：城市等） |
| `Actor1Geo_FullName` | `varchar` | 參與者 1 所在地的完整地理名稱 |
| `Actor1Geo_CountryCode` | `varchar` | 參與者 1 所在地的國家代碼 |
| `Actor1Geo_ADM1Code` | `varchar` | 參與者 1 所在地的第一級行政區代碼 |
| `Actor1Geo_ADM2Code` | `varchar` | 參與者 1 所在地的第二級行政區代碼 |
| `Actor1Geo_Lat` | `double` | 參與者 1 所在地的緯度 |
| `Actor1Geo_Long` | `double` | 參與者 1 所在地的經度 |
| `Actor1Geo_FeatureID` | `varchar` | 參與者 1 所在地的 GNS／GNIS 地理特徵代碼 |
| `Actor2Geo_Type` ～ `Actor2Geo_FeatureID` | `mixed` | 參與者 2 的地理位置資訊（欄位結構同 `Actor1Geo`） |
| `ActionGeo_Type` ～ `ActionGeo_FeatureID` | `mixed` | 事件實際發生地的地理位置資訊（欄位結構同 `Actor1Geo`） |
| `DATEADDED` | `int64` | 資料新增至系統的時間戳記（格式：`YYYYMMDDHHMMSS`） |
| `SOURCEURL` | `varchar` | 報導此事件的原始新聞網址（可點擊查看原文） |
