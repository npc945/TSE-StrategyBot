# 台股多因子選股與自動化推播交易系統

>  **專案簡介**
> 本專案為一套End-to-End的台股量化交易系統。整合了 **FinMind API 爬蟲、MySQL 資料庫、Pandas 特徵工程、LSTM 深度學習模型預測、以及 Streamlit 視覺化儀表板**。透過嚴謹的程式化回測與每日自動化排程，解決散戶情緒化交易的痛點，實現具備高度風險控管 的量化交易策略。

## 技術與量化成果
* **數據工程與特徵處理**
  * 利用 `pandas` 與 `pandas-ta` 處理台股歷史價量資料，進行缺失值處理與異常值(資料區間為2015/1/1~2025/10/23)，並透過時間序列平移 (Shift) 防止洩漏造成模型準確度偏差。
  * 自主開發 MySQL 資料庫的 `Upsert` 機制，每日盤後自動化增量更新數據，大幅降低 API 請求次數與運算開銷。
* **AI 模型濾網**
  * 針對台積電 (2330)、聯發科 (2454)、鴻海 (2317) 建立專屬 LSTM 預測模型。
  * 結合 ADX 趨勢強度與 SMA 均線作為技術濾網，有效抑制偽陽性 (False Positive) 買進訊號。回測數據顯示，進場訊號精確率 (Precision) **穩定提升至 64% ~ 75%**。
* **風險控管成效**
  * 經 2023-2025 年留存測試集驗證，雙軌策略成功避開多次大盤主跌段，將投資組合之**最大回撤 (MDD) 較單純持有策略大幅縮減近 50%**。
* **系統自動化與部署**
  * 開發自動化腳本，每日盤後無縫銜接「資料抓取 -> 指標運算 -> 模型推論 -> 訊號生成」。
  * 建立基於 `Streamlit` 的互動式 Web 儀表板，並串接 LLM API 生成即時自然語言解盤報告。

---

## 專案架構與模組說明 

為了達到高內聚、低耦合的軟體工程標準，本系統將各功能拆分為獨立模組：

### 1. 資料工程層 
* `stock_finmind.py`：負責串接 FinMind API，執行每日最新價量資料的爬取與增量更新。
* `sql_upsert.py`：封裝 MySQL 資料庫連線與 `ON DUPLICATE KEY UPDATE` (Upsert) 邏輯，確保資料唯一性。
* `tech.py`：技術指標運算引擎。使用 Pandas 讀取原始數據，計算 SMA、ADX、RSI、BIAS 等多維度特徵矩陣，並寫回資料庫供模型使用。

### 2. AI 模型層
* `LSTM2330.py` / `LSTM2454.py` / `LSTM2317.py`：分別針對三大核心權值股建構的 TensorFlow/Keras LSTM 模型訓練與預測腳本，包含資料標準化 (MinMaxScaler) 與時間序列切割 (TimeSeriesSplit)。

### 3. 量化回測與策略
* `backtest.py`：AI 核心策略的回測引擎，計算整合 LSTM 預測與技術濾網後的歷史回測績效 (Sharpe Ratio, MDD, Win Rate)。
* `control_backtest.py`：純技術分析的選擇，讓使用者能夠有多樣化的選擇。
* `daily_test.py`：每日實盤自動化推論腳本。讀取當日最新特徵，餵入預訓練模型產出預測機率，並輸出最終的買賣交易訊號。

### 4. 終端應用層
*  `web.py`：基於 Streamlit 開發的視覺化量化分析儀表板。提供資金模擬、歷史 K 線動態追蹤、AI 模型精度漏斗圖及交易明細檢視。

### 5. 環境設定
*  `requirements.txt`：專案依賴套件清單 (包含 `pandas`, `numpy`, `tensorflow`, `streamlit`, `SQLAlchemy` 等)。
* `.devcontainer/` & `.streamlit/`：開發環境與網頁部署設定檔。

---

## 技術

* **開發語言:** Python 3.10+
* **資料處理:** Pandas, NumPy, Pandas-TA
* **機器學習:** TensorFlow, Keras, Scikit-learn
* **資料庫:** MySQL, SQLAlchemy (ORM)
* **視覺化與前端:** Streamlit, Matplotlib, Plotly
* **版本控制與自動化:** Git, GitHub Actions / Windows Task Scheduler

## 未來優化方向 
1. 串接國內券商實盤 API，達成全自動程式化下單。
2. 導入總體經濟數據與財經新聞 NLP 情緒分析，擴充特徵矩陣維度。
3. 將系統容器化 (Docker)，提升部署至雲端伺服器 (AWS/GCP) 的便利性與穩定性。