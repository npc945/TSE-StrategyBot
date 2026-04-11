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
# ⚙️ 1. 環境設定與股票參數
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "token.env")
load_dotenv(env_path)
sql_engine_url = os.getenv('sql_engine')

LOOK_BACK = 20
TOTAL_CAPITAL = 1000000  # 固定本金 100 萬

# 手續費設定（台股）
BUY_FEE  = 0.001425
SELL_FEE = 0.001425 + 0.003

TEST_START = "2023-08-08"
TEST_END   = "2025-10-25"

STOCK_CONFIG = {
    "2317": {"features": ["Trading_Volume", "Bias_20"], "ai_threshold": 0.5},
    "2330": {"features": ["Trading_Volume", "Bias_20"], "ai_threshold": 0.5},
    "2454": {"features": ["RSI_14", "Bias_20"],         "ai_threshold": 0.5}
}

engine = create_engine(sql_engine_url)

# ==========================================
# 🚦 2. 策略濾網函數
# ==========================================
def get_lights(data, i, ai_threshold):
    l = 0
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
# 🚀 3. 自動化巡迴回測主引擎
# ==========================================
for STOCK_ID, config in STOCK_CONFIG.items():
    print(f"\n{'='*50}")
    print(f"🔄 正在執行專業回測：{STOCK_ID}")
    print(f"📅 測試區間：{TEST_START} ~ {TEST_END}")
    print(f"{'='*50}")

    AI_FEATURES  = config["features"]
    AI_THRESHOLD = config["ai_threshold"]

    try:
        model  = load_model(f"{STOCK_ID}_model.h5")
        scaler = joblib.load(f"{STOCK_ID}_scaler.pkl")
    except Exception as e:
        print(f"❌ 找不到 {STOCK_ID} 的模型檔案，跳過。錯誤: {e}")
        continue

    df_all = pd.read_sql(f"SELECT * FROM stock_data WHERE stock_id={STOCK_ID} ORDER BY date", engine)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['SMA_Vol_20'] = df_all['Trading_Volume'].rolling(20).mean()

    try:
        start_idx = df_all[df_all['date'] >= TEST_START].index[0]
        end_df    = df_all[df_all['date'] <= TEST_END]
        end_idx   = end_df.index[-1]
    except IndexError:
        print(f"❌ 找不到 {STOCK_ID} 指定的日期區間，跳過。")
        continue

    df_test_period = df_all.iloc[start_idx - LOOK_BACK: end_idx + 1].copy().reset_index(drop=True)

    X_raw    = df_test_period[AI_FEATURES].values
    X_scaled = scaler.transform(X_raw)
    Xs       = [X_scaled[i: i + LOOK_BACK] for i in range(len(X_scaled) - LOOK_BACK)]
    X_3d     = np.array(Xs)

    probs  = model.predict(X_3d, verbose=0).flatten()
    df_res = df_test_period.iloc[LOOK_BACK:].copy().reset_index(drop=True)
    df_res['AI_Prob'] = probs

    # 回測變數
    holding_qty      = 0
    actual_buy_price = 0
    total_profit     = 0
    realized_profit  = 0
    trades           = []
    daily_values     = []

    for i in range(len(df_res)):
        lights      = get_lights(df_res, i, AI_THRESHOLD)
        is_last_day = (i == len(df_res) - 1)

        if holding_qty == 0 and not is_last_day:
            if lights >= 2:
                weight           = 1.0 if lights >= 3 else 0.5
                invest_amount    = TOTAL_CAPITAL * weight
                raw_price        = df_res['close'].iloc[i]
                actual_buy_price = raw_price * (1 + BUY_FEE)
                holding_qty      = int(invest_amount / actual_buy_price)

                trades.append({
                    'date'       : df_res['date'].iloc[i].strftime('%Y-%m-%d'),
                    'action'     : 'BUY',
                    'qty'        : holding_qty,
                    'price'      : round(float(raw_price), 2),
                    'actual_cost': round(float(actual_buy_price), 2),
                    'profit'     : 0.0,
                    'profit_pct' : 0.0
                })

        elif holding_qty > 0:
            if check_exit(df_res, i) or is_last_day:
                raw_sell    = df_res['close'].iloc[i]
                actual_sell = raw_sell * (1 - SELL_FEE)
                profit      = (actual_sell - actual_buy_price) * holding_qty
                profit_pct  = ((actual_sell - actual_buy_price) / actual_buy_price) * 100
                realized_profit += profit
                total_profit    += profit
                action_name = 'SELL (結算)' if is_last_day else 'SELL'

                trades.append({
                    'date'       : df_res['date'].iloc[i].strftime('%Y-%m-%d'),
                    'action'     : action_name,
                    'qty'        : holding_qty,
                    'price'      : round(float(raw_sell), 2),
                    'actual_cost': round(float(actual_sell), 2),
                    'profit'     : round(float(profit), 2),
                    'profit_pct' : round(profit_pct, 2)
                })
                holding_qty = 0

        if holding_qty > 0:
            est_sell  = df_res['close'].iloc[i] * (1 - SELL_FEE)
            day_value = TOTAL_CAPITAL + realized_profit + (est_sell - actual_buy_price) * holding_qty
        else:
            day_value = TOTAL_CAPITAL + realized_profit
        daily_values.append(day_value)

    # ==========================================
    # 4. 績效指標計算
    # ==========================================
    trade_df    = pd.DataFrame(trades)
    sell_trades = trade_df[trade_df['action'].str.contains('SELL')] if not trade_df.empty else pd.DataFrame()
    win_rate    = (len(sell_trades[sell_trades['profit'] > 0]) / len(sell_trades) * 100) if len(sell_trades) > 0 else 0.0
    total_profit_pct = (total_profit / TOTAL_CAPITAL) * 100

    # 策略夏普比率
    daily_series  = pd.Series(daily_values)
    daily_returns = daily_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0

    # 策略最大回撤
    rolling_max  = daily_series.cummax()
    drawdown     = (daily_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # 同期單純持有：夏普、報酬率、最大回撤
    hold_returns    = df_res['close'].pct_change().dropna()
    hold_sharpe     = (hold_returns.mean() / hold_returns.std()) * np.sqrt(252) if hold_returns.std() > 0 else 0.0
    hold_return_pct = ((df_res['close'].iloc[-1] - df_res['close'].iloc[0]) / df_res['close'].iloc[0]) * 100

    # 單純持有最大回撤（從第一天買進一路抱著不動）
    hold_series      = df_res['close'] / df_res['close'].iloc[0]   # 標準化成從 1 開始
    hold_rolling_max = hold_series.cummax()
    hold_drawdown    = (hold_series - hold_rolling_max) / hold_rolling_max
    hold_max_drawdown = hold_drawdown.min() * 100

    print(f"總投入本金上限      : ${TOTAL_CAPITAL:,.0f}")
    print(f"累積淨利（含手續費）: ${total_profit:,.0f}")
    print(f"總報酬率            : {total_profit_pct:.2f}%")
    print(f"勝率                : {win_rate:.2f}% | 交易次數: {len(sell_trades)} 次")
    print(f"────────────────────────────────")
    print(f"策略夏普比率（年化）: {sharpe:.4f}")
    print(f"持有夏普比率（年化）: {hold_sharpe:.4f}  ← 同期單純持有")
    print(f"────────────────────────────────")
    print(f"策略最大回撤        : {max_drawdown:.2f}%")
    print(f"持有最大回撤        : {hold_max_drawdown:.2f}%  ← 同期單純持有")
    print(f"────────────────────────────────")
    print(f"策略總報酬率        : {total_profit_pct:.2f}%")
    print(f"單純持有報酬率      : {hold_return_pct:.2f}%  ← 同期單純持有")

    #  5. 匯出 JSON / CSV
    kpi_data = {
        "stock_id"             : STOCK_ID,
        "max_capital"          : float(TOTAL_CAPITAL),
        "total_profit"         : round(float(total_profit), 2),
        "total_return_pct"     : round(total_profit_pct, 2),
        "win_rate"             : round(win_rate, 2),
        "total_trades"         : len(sell_trades),
        "sharpe_ratio"         : round(sharpe, 4),
        "max_drawdown_pct"     : round(max_drawdown, 2),
        "hold_sharpe_ratio"    : round(hold_sharpe, 4),
        "hold_return_pct"      : round(hold_return_pct, 2),
        "hold_max_drawdown_pct": round(hold_max_drawdown, 2)
    }

    export_data = {
        "kpi"   : kpi_data,
        "trades": trade_df.to_dict(orient='records') if not trade_df.empty else []
    }

    with open(f"web_data_{STOCK_ID}_proof.json", 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=4)

    df_kline = df_res[['date', 'open', 'high', 'low', 'close', 'SMA_20']].copy()
    df_kline['date'] = df_kline['date'].dt.strftime('%Y-%m-%d')
    df_kline.to_csv(f"web_kline_{STOCK_ID}.csv", index=False)

    print(f"✅ 成功匯出：web_data_{STOCK_ID}_proof.json 與 web_kline_{STOCK_ID}.csv")

print("\n🚀 所有標的回測完成（含手續費、夏普比率、單純持有最大回撤比較），證明檔與 K 線檔皆已更新完畢！")