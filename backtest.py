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
# ⚙️ 1. 設定區
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "token.env")
load_dotenv(env_path)
sql_engine_url = os.getenv('sql_engine')
STOCK_ID = 2317
LOOK_BACK = 20
AI_FEATURES = ["Trading_Volume", "Bias_20"]
TRAIN_RATIO = 0.8  # 🌟 設定訓練集比例，回測將從後 20% 開始

# ==========================================
# 🧠 2. 載入模型與 Scaler
# ==========================================
model = load_model(f"{STOCK_ID}_model.h5")
scaler = joblib.load(f"{STOCK_ID}_scaler.pkl")

# ==========================================
# 📊 3. 讀取 SQL 並切分「測試集」
# ==========================================
engine = create_engine(sql_engine_url)
df_all = pd.read_sql(f"SELECT * FROM stock_data WHERE stock_id={STOCK_ID} ORDER BY date", engine)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['SMA_Vol_20'] = df_all['Trading_Volume'].rolling(20).mean()

# 🎯 關鍵：找出測試集的起始索引
split_idx = int(len(df_all) * TRAIN_RATIO)
# 回測資料需要包含測試集前 LOOK_BACK 天，AI 才能計算第一筆預測
df_test_period = df_all.iloc[split_idx - LOOK_BACK :].copy().reset_index(drop=True)

print(f"📈 回測區間：{df_test_period['date'].iloc[LOOK_BACK].date()} 至今 (測試集範圍)")

# ==========================================
# 🔮 4. AI 預測區 (僅針對測試集)
# ==========================================
X_raw = df_test_period[AI_FEATURES].values
X_scaled = scaler.transform(X_raw)

Xs = []
for i in range(len(X_scaled) - LOOK_BACK):
    Xs.append(X_scaled[i : i + LOOK_BACK])
X_3d = np.array(Xs)

probs = model.predict(X_3d, verbose=0).flatten()

# 對齊：去掉前面的 LOOK_BACK 天，這才是真正的測試集起點
df_res = df_test_period.iloc[LOOK_BACK:].copy().reset_index(drop=True)
df_res['AI_Prob'] = probs
print("🧠 AI 預測機率分佈：")
print(df_res['AI_Prob'].describe())
# ==========================================
# 🚦 5. 五燈獎與賣出邏輯 (沿用你的精密設計)
# ==========================================
def get_lights(data, i):
    l = 0
    if data['AI_Prob'].iloc[i] > 0.5: l += 1
    if data['ADX_14'].iloc[i] > 20: l += 1
    if i > 0:
        if data['close'].iloc[i] > data['SMA_20'].iloc[i] and data['SMA_20'].iloc[i] > data['SMA_20'].iloc[i-1]:
            l += 1
        if data['low'].iloc[i] > data['high'].iloc[i-1] and data['close'].iloc[i] > data['open'].iloc[i]:
            l += 1
    # L5: 低位放量 (檢查當前索引在原始資料中的回測)
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
#  6. 模擬回測執行 (自動算本金版)
# ==========================================
holding = 0
buy_price = 0
total_profit = 0
trades = []

max_capital_used = 0 # 新增：用來記錄這套策略「到底花了多少錢」

for i in range(len(df_res)):
    lights = get_lights(df_res, i)
    is_last_day = (i == len(df_res) - 1)
    
    # 買進邏輯
    if holding == 0 and not is_last_day:
        if lights >= 3:
            holding = 2
            buy_price = df_res['close'].iloc[i]
            
            # 🌟 計算這次花了多少錢，如果破紀錄就存起來
            current_cost = buy_price * holding * 1000 
            if current_cost > max_capital_used: 
                max_capital_used = current_cost
                
            trades.append({'date': df_res['date'].iloc[i].strftime('%Y-%m-%d'), 'action': 'BUY', 'qty': 2, 'price': buy_price, 'profit': 0, 'profit_pct': 0.0})
            
        elif lights == 2:
            holding = 1
            buy_price = df_res['close'].iloc[i]
            
            # 🌟 計算這次花了多少錢，如果破紀錄就存起來
            current_cost = buy_price * holding * 1000 
            if current_cost > max_capital_used: 
                max_capital_used = current_cost
                
            trades.append({'date': df_res['date'].iloc[i].strftime('%Y-%m-%d'), 'action': 'BUY', 'qty': 1, 'price': buy_price, 'profit': 0, 'profit_pct': 0.0})

    # 賣出邏輯
    elif holding > 0:
        if check_exit(df_res, i) or is_last_day:
            sell_price = df_res['close'].iloc[i]
            
            profit = (sell_price - buy_price) * holding * 1000
            profit_pct = ((sell_price - buy_price) / buy_price) * 100 
            
            total_profit += profit
            action_name = 'SELL (結算)' if is_last_day else 'SELL'
            
            trades.append({'date': df_res['date'].iloc[i].strftime('%Y-%m-%d'), 'action': action_name, 'qty': holding, 'price': sell_price, 'profit': profit, 'profit_pct': round(profit_pct, 2)})
            holding = 0

# ==========================================
#  7. 成果結算
# ==========================================
trade_df = pd.DataFrame(trades)

# 自動計算總報酬率：用「實際動用到的最大資金」當作分母！一毛錢都不吃虧！
if max_capital_used > 0:
    total_profit_pct = (total_profit / max_capital_used) * 100
else:
    total_profit_pct = 0.0

print("\n" + "-" * 20)
print(f"📊 {STOCK_ID} 測試集回測報告 (最大動用資金法)")
print(f"💵 策略最高動用資金: ${max_capital_used:,.0f}")
print(f"💰 累積總盈虧: ${total_profit:,.0f}")
print(f"🚀 策略總報酬率: {total_profit_pct:.2f}%") # 這就是你最完美、沒有被稀釋的 % 數！

if not trade_df.empty and 'profit' in trade_df.columns:
    sell_trades = trade_df[trade_df['action'].str.contains('SELL')]
    if len(sell_trades) > 0:
        win_rate = (len(sell_trades[sell_trades['profit'] > 0]) / len(sell_trades) * 100)
        print(f"🎯 勝率: {win_rate:.2f}% | 交易次數: {len(sell_trades)} 次")

print("-" * 20)
print(trade_df.to_string(index=False))

# ==========================================
# 📦 8. 匯出給網頁使用的 JSON 與 CSV 檔案
# ==========================================
import json

# 1. 整理 KPI 數據
kpi_data = {
    "stock_id": STOCK_ID,
    "max_capital": float(max_capital_used),
    "total_profit": float(total_profit),
    "total_return_pct": round(total_profit_pct, 2),
    "win_rate": round(win_rate, 2) if 'win_rate' in locals() else 0.0,
    "total_trades": len(sell_trades) if 'sell_trades' in locals() else 0
}

# 2. 轉換交易明細
trades_list = trade_df.to_dict(orient='records')
export_data = {"kpi": kpi_data, "trades": trades_list}

# # 3. 儲存 JSON (KPI 與 交易明細)
# json_filename = f"web_data_{STOCK_ID}_proof.json"
# with open(json_filename, 'w', encoding='utf-8') as f:
#     json.dump(export_data, f, ensure_ascii=False, indent=4)

# # 4. 🌟 新增：儲存測試集的 K 線資料 (給 Plotly 畫圖用)
# # 只抓取畫圖需要的欄位，減輕網頁負擔
# df_kline = df_res[['date', 'open', 'high', 'low', 'close', 'SMA_20']].copy()
# # 將日期格式化為字串，避免 JSON/CSV 讀取錯誤
# df_kline['date'] = df_kline['date'].dt.strftime('%Y-%m-%d')
# csv_filename = f"web_kline_{STOCK_ID}.csv"
# df_kline.to_csv(csv_filename, index=False)

# print(f"✅ 網頁展示資料已成功匯出：{json_filename} 與 {csv_filename}")