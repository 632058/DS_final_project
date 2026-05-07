# Feature Engineering

> Owner: 成員 B

## 1. 特徵設計動機

[為什麼用 rolling、lag、diff(前兆訊號需要時間結構)]

## 2. 缺值處理策略

[count 類補 0、average 類用 ffill 的理由]

## 3. 時間連續性處理

[每國從第一週到最新週皆有列,缺週補齊]

## 4. 避免 look-ahead bias

[所有 rolling 都 shift(1)、所有 lag 都從 1 起跳]

## 5. Pooling 策略

[country one-hot 的目的:樣本不足要 pooling]

## 6. 完整特徵清單

[用表格列出所有產出特徵]
