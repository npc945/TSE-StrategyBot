import pandas as pd
import numpy as np
import os
import joblib
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import warnings
import json
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ 1. 環境設定與股票字典 (Configuration)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "token.env")
load_dotenv(env_path)
sql_engine_url = os.getenv('sql_engine')

LOOK_BACK = 20

# 🌟 核心引擎設定檔：一次控制三檔股票的所有差異
STOCK_CONFIG = {
    "2317": {
        "features": ["Trading_Volume", "Bias_20"],
        "ai_threshold": 0.5,
        "test_start": "2023-08-08"  # ⚠️ 請改成你 2317 測試集真正的第一天
    },
    "2330": {
        "features": ["Trading_Volume", "Bias_20"],
        "ai_threshold": 0.5,
        "test_start": "2023-08-08"  # ⚠️ 請改成你 2330 測試集真正的第一天
    },
    "2454": {
        "features": ["RSI_14", "Bias_20"],
        "ai_threshold": 0.6,        # 🌟 專屬的 0.6 買進閾值
        "test_start": "2023-08-08"  # ⚠️ 請改成你 2454 測試集真正的第一天
    }
}

engine = create_engine(sql_engine_url)

# ==========================================
# 🚦 2. 策略濾網函數 (加入動態閾值參數)
# ==========================================
def get_lights(data, i, ai_threshold):
    l = 0
    # 🌟 AI 機率超過專屬閾值才亮燈
    if data['AI_Prob'].iloc[i] > ai_threshold: l += 1
    if data['ADX_14'].iloc[i] > 20: l += 1
    if i > 0:
        if data['close'].iloc[i] > data['SMA_20'].iloc[i] and data['SMA_20'].iloc[i] > data['SMA_20'].iloc[i-1]:
            l += 1
        if data['low'].iloc[i] > data['high'].iloc[i-1] and data['close'].iloc[i] > data['open'].iloc[i]:
            l += 1
    has_low_bias = (data['Bias_20'].iloc[max(0, i-20):i] < -0.05).any()
    vol_3d_12 = (data['Trading_Volume'].iloc[max(0, i-2):i+1] > data['SMA_Vol_20'].iloc[i] * 1.2).all()
    vol_1d_20 = data['Trading_Volume'].iloc[i] > data['SMA_Vol_20'].iloc[i] * 2.0
    if has_low_bias and (vol_3d_12 or vol_1d_20):
        l += 1
    return l

def check_exit(data, i):
    if i < 1: return False
    exit_lights = 0
    if data['close'].iloc[i] < data['SMA_20'].iloc[i] and data['SMA_20'].iloc[i] < data['SMA_20'].iloc[i-1]:
        exit_lights += 1
    if data['ADX_14'].iloc[i] > 20:
        exit_lights += 1
    if data['high'].iloc[i] < data['low'].iloc[i-1] and data['close'].iloc[i] < data['open'].iloc[i]:
        exit_lights += 1
    return exit_lights >= 2

# ==========================================
# 🚀 3. 自動化巡迴主引擎 (Main Loop)
# ==========================================
for STOCK_ID, config in STOCK_CONFIG.items():
    print(f"\n{'='*40}")
    print(f"🔄 正在執行每日跟進：{STOCK_ID}")
    print(f"{'='*40}")
    
    # 讀取專屬設定
    AI_FEATURES = config["features"]
    AI_THRESHOLD = config["ai_threshold"]
    TEST_START = config["test_start"]
    
    # 載入模型與 Scaler
    try:
        model = load_model(f"{STOCK_ID}_model.h5")
        scaler = joblib.load(f"{STOCK_ID}_scaler.pkl")
    except Exception as e:
        print(f"❌ 找不到 {STOCK_ID} 的模型或 Scaler 檔案，跳過此檔。錯誤: {e}")
        continue

    # 讀取 SQL
    df_all = pd.read_sql(f"SELECT * FROM stock_data WHERE stock_id={STOCK_ID} ORDER BY date", engine)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['SMA_Vol_20'] = df_all['Trading_Volume'].rolling(20).mean()

    # 🎯 鎖死測試集起點 (避免歷史位移)
    try:
        start_idx = df_all[df_all['date'] >= TEST_START].index[0]
    except IndexError:
        print(f"❌ 找不到 {STOCK_ID} 設定的起始日期 {TEST_START}，跳過。")
        continue
        
    df_test_period = df_all.iloc[start_idx - LOOK_BACK :].copy().reset_index(drop=True)

    # AI 預測
    X_raw = df_test_period[AI_FEATURES].values
    X_scaled = scaler.transform(X_raw)
    Xs = [X_scaled[i : i + LOOK_BACK] for i in range(len(X_scaled) - LOOK_BACK)]
    X_3d = np.array(Xs)

    probs = model.predict(X_3d, verbose=0).flatten()

    df_res = df_test_period.iloc[LOOK_BACK:].copy().reset_index(drop=True)
    df_res['AI_Prob'] = probs

    # 模擬回測執行
    holding = 0
    buy_price = 0
    total_profit = 0
    trades = []
    max_capital_used = 0 

    for i in range(len(df_res)):
        lights = get_lights(df_res, i, AI_THRESHOLD)
        
        # 買進邏輯
        if holding == 0 and i != len(df_res) - 1: # 今天如果是空手，就不在最後一刻買，等明天確認
            if lights >= 3:
                holding = 2
                buy_price = df_res['close'].iloc[i]
                current_cost = buy_price * holding * 1000 
                if current_cost > max_capital_used: max_capital_used = current_cost
                trades.append({'date': df_res['date'].iloc[i].strftime('%Y-%m-%d'), 'action': 'BUY', 'qty': 2, 'price': float(buy_price), 'profit': 0.0, 'profit_pct': 0.0})
                
            elif lights == 2:
                holding = 1
                buy_price = df_res['close'].iloc[i]
                current_cost = buy_price * holding * 1000 
                if current_cost > max_capital_used: max_capital_used = current_cost
                trades.append({'date': df_res['date'].iloc[i].strftime('%Y-%m-%d'), 'action': 'BUY', 'qty': 1, 'price': float(buy_price), 'profit': 0.0, 'profit_pct': 0.0})

        # 賣出與持倉邏輯
        elif holding > 0:
            # 🌟 取消強制結算，只有真正滿足賣出條件才賣
            if check_exit(df_res, i):
                sell_price = df_res['close'].iloc[i]
                profit = (sell_price - buy_price) * holding * 1000
                profit_pct = ((sell_price - buy_price) / buy_price) * 100 
                total_profit += profit
                trades.append({'date': df_res['date'].iloc[i].strftime('%Y-%m-%d'), 'action': 'SELL', 'qty': holding, 'price': float(sell_price), 'profit': float(profit), 'profit_pct': round(profit_pct, 2)})
                holding = 0
                
            # 🌟 如果今天是最後一天，且沒有賣出，則計算「未實現損益」
            elif i == len(df_res) - 1:
                current_price = df_res['close'].iloc[i]
                unrealized_profit = (current_price - buy_price) * holding * 1000
                unrealized_pct = ((current_price - buy_price) / buy_price) * 100
                total_profit += unrealized_profit
                trades.append({'date': df_res['date'].iloc[i].strftime('%Y-%m-%d'), 'action': 'HOLDING (持倉中)', 'qty': holding, 'price': float(current_price), 'profit': float(unrealized_profit), 'profit_pct': round(unrealized_pct, 2)})
                print(f"📌 {STOCK_ID} 狀態：持倉中！目前帳面盈虧 ${unrealized_profit:,.0f}")

    # 成果結算與匯出
    trade_df = pd.DataFrame(trades)
    total_profit_pct = (total_profit / max_capital_used * 100) if max_capital_used > 0 else 0.0

    sell_trades = trade_df[trade_df['action'] == 'SELL'] if not trade_df.empty else pd.DataFrame()
    win_rate = (len(sell_trades[sell_trades['profit'] > 0]) / len(sell_trades) * 100) if len(sell_trades) > 0 else 0.0

    print(f"💰 {STOCK_ID} 累積淨利: ${total_profit:,.0f} | 總報酬率: {total_profit_pct:.2f}% | 勝率: {win_rate:.2f}%")

    # 📦 匯出檔案給網頁使用 (命名加上 _daily 區別)
    kpi_data = {
        "stock_id": STOCK_ID,
        "max_capital": float(max_capital_used),
        "total_profit": float(total_profit),
        "total_return_pct": round(total_profit_pct, 2),
        "win_rate": round(win_rate, 2),
        "total_trades": len(sell_trades),
        "current_status": "HOLDING" if holding > 0 else "EMPTY"
    }

    export_data = {"kpi": kpi_data, "trades": trade_df.to_dict(orient='records') if not trade_df.empty else []}
    
    json_filename = f"web_data_{STOCK_ID}_daily.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=4)

    df_kline = df_res[['date', 'open', 'high', 'low', 'close', 'SMA_20']].copy()
    df_kline['date'] = df_kline['date'].dt.strftime('%Y-%m-%d')
    df_kline.to_csv(f"web_kline_{STOCK_ID}_daily.csv", index=False)

print("\n✅ 所有標的每日跟進與 JSON/CSV 更新完畢！")