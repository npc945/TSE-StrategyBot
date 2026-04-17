import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import warnings
import json
warnings.filterwarnings('ignore')

#LINE
try:
    from line import send_line_message
except ImportError:
    print("⚠️ 找不到 line.py 模組，LINE 通知將被忽略。")
    def send_line_message(msg): pass

def generate_trade_signal_msg(date, stock_id, action, price, is_ai_stock):
    # 建立股票名稱字典，讓 LINE 顯示更好看
    name_dict = {"2330": "台積電", "2454": "聯發科", "2317": "鴻海", "2382": "廣達", "2308": "台達電", "2881": "富邦金", "2603": "長榮", "1301": "台塑", "1513": "中興電", "2412": "中華電"}
    stock_name = name_dict.get(stock_id, "台股")

    if is_ai_stock:
        strategy_title = "🧠 【AI 核心策略訊號】"
        ai_warning = "" 
    else:
        strategy_title = "⚙️ 【純技術分析策略訊號】"
        ai_warning = "\n⚠️ 系統提示：此檔標的為對照組，『並未』加入 LSTM AI 預測模型。\n----------------------"

    if action == "BUY":
        action_title = "🟢 買進 (BUY)"
        action_advice = (
            "📝 實單操作建議：\n"
            "若欲跟單，請於『下個交易日開盤前』，掛【今日收盤價位】買進。\n"
            "💡 若開盤沒有順利成交，代表股價已跳空，建議【不要追高入場】以避免滑價。"
        )
    elif action == "SELL":
        action_title = "🔴 賣出 (SELL)"
        action_advice = (
            "📝 實單操作建議：\n"
            "請於『下個交易日開盤前』，掛【今日收盤價位】賣出。\n"
            "💡 若開盤未成交，建議務必於【交易日收盤前出清】。"
        )
    else:
        return None 

    msg = f"""{strategy_title}
📅 日期：{date}
----------------------
🔎 標的：{stock_name} ({stock_id})
🔔 動作：{action_title}
💰 訊號價位：${price:,.2f}
----------------------{ai_warning}
{action_advice}
----------------------
⚠️ 免責聲明：
本系統訊號僅供歷史回測與學術參考。就算跟單還是會有市場波動風險，請投資人自行評估並謹慎操作。"""
    return msg

# ==========================================
# ⚙️ 1. 環境設定與對照組股票字典
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "token.env")
load_dotenv(env_path)
sql_engine_url = os.getenv('sql_engine')

LOOK_BACK = 20
TOTAL_CAPITAL = 1000000  # 本金100萬

# 手續費設定（台股）
BUY_FEE  = 0.001425
SELL_FEE = 0.001425 + 0.003

STOCK_CONFIG = {
    "2382": {"test_start": "2023-08-08"},
    "2308": {"test_start": "2023-08-08"},
    "2881": {"test_start": "2023-08-08"},
    "2603": {"test_start": "2023-08-08"},
    "1301": {"test_start": "2023-08-08"},
    "1513": {"test_start": "2023-08-08"},
    "2412": {"test_start": "2023-08-08"}
}

engine = create_engine(sql_engine_url)

# ==========================================
# 🚦 2. 純技術面策略濾網函數
# ==========================================
def get_lights(data, i):
    l = 0
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
# 🚀 3. 對照組自動化巡迴主引擎
# ==========================================
for STOCK_ID, config in STOCK_CONFIG.items():
    print(f"\n{'='*40}")
    print(f"🔄 正在執行傳統對照組回測：{STOCK_ID}")
    print(f"{'='*40}")

    TEST_START = config["test_start"]

    df_all = pd.read_sql(f"SELECT * FROM stock_data WHERE stock_id={STOCK_ID} ORDER BY date", engine)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['SMA_Vol_20'] = df_all['Trading_Volume'].rolling(20).mean()

    try:
        start_idx = df_all[df_all['date'] >= TEST_START].index[0]
    except IndexError:
        print(f"❌ 找不到 {STOCK_ID} 的資料或日期 {TEST_START}，請確認資料庫。")
        continue

    df_test_period = df_all.iloc[start_idx - LOOK_BACK:].copy().reset_index(drop=True)
    df_res = df_test_period.iloc[LOOK_BACK:].copy().reset_index(drop=True)

    # 回測主迴圈
    holding_qty      = 0
    actual_buy_price = 0
    total_profit     = 0
    realized_profit  = 0
    trades           = []
    daily_values     = []

    for i in range(len(df_res)):
        lights = get_lights(df_res, i)

        if holding_qty == 0:
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
            if check_exit(df_res, i):
                raw_sell    = df_res['close'].iloc[i]
                actual_sell = raw_sell * (1 - SELL_FEE)
                profit      = (actual_sell - actual_buy_price) * holding_qty
                profit_pct  = ((actual_sell - actual_buy_price) / actual_buy_price) * 100
                realized_profit += profit
                total_profit    += profit

                trades.append({
                    'date'       : df_res['date'].iloc[i].strftime('%Y-%m-%d'),
                    'action'     : 'SELL',
                    'qty'        : holding_qty,
                    'price'      : round(float(raw_sell), 2),
                    'actual_cost': round(float(actual_sell), 2),
                    'profit'     : round(float(profit), 2),
                    'profit_pct' : round(profit_pct, 2)
                })
                holding_qty = 0

            elif i == len(df_res) - 1:
                current_price     = df_res['close'].iloc[i]
                actual_sell_now   = current_price * (1 - SELL_FEE)
                unrealized_profit = (actual_sell_now - actual_buy_price) * holding_qty
                unrealized_pct    = ((actual_sell_now - actual_buy_price) / actual_buy_price) * 100
                total_profit     += unrealized_profit

                trades.append({
                    'date'       : df_res['date'].iloc[i].strftime('%Y-%m-%d'),
                    'action'     : 'HOLDING',
                    'qty'        : holding_qty,
                    'price'      : round(float(current_price), 2),
                    'actual_cost': round(float(actual_sell_now), 2),
                    'profit'     : round(float(unrealized_profit), 2),
                    'profit_pct' : round(unrealized_pct, 2)
                })
                print(f"📌 {STOCK_ID} 狀態：持倉中！帳面盈虧（含手續費）${unrealized_profit:,.0f}")

        if holding_qty > 0:
            est_sell  = df_res['close'].iloc[i] * (1 - SELL_FEE)
            day_value = TOTAL_CAPITAL + realized_profit + (est_sell - actual_buy_price) * holding_qty
        else:
            day_value = TOTAL_CAPITAL + realized_profit
        daily_values.append(day_value)

    # ==========================================
    # 📊 4. 績效指標計算
    # ==========================================
    trade_df    = pd.DataFrame(trades)
    sell_trades = trade_df[trade_df['action'] == 'SELL'] if not trade_df.empty else pd.DataFrame()
    win_rate    = (len(sell_trades[sell_trades['profit'] > 0]) / len(sell_trades) * 100) if len(sell_trades) > 0 else 0.0
    total_profit_pct = (total_profit / TOTAL_CAPITAL) * 100

    # 策略夏普比率
    daily_series  = pd.Series(daily_values)
    daily_returns = daily_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0

    # 最大回撤
    rolling_max  = daily_series.cummax()
    drawdown     = (daily_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # 🌟 同期單純持有：夏普、報酬率、最大回撤 (這裡幫你補齊了！)
    hold_returns    = df_res['close'].pct_change().dropna()
    hold_sharpe     = (hold_returns.mean() / hold_returns.std()) * np.sqrt(252) if hold_returns.std() > 0 else 0.0
    hold_return_pct = ((df_res['close'].iloc[-1] - df_res['close'].iloc[0]) / df_res['close'].iloc[0]) * 100
    
    hold_series       = df_res['close'] / df_res['close'].iloc[0]
    hold_rolling_max  = hold_series.cummax()
    hold_drawdown     = (hold_series - hold_rolling_max) / hold_rolling_max
    hold_max_drawdown = hold_drawdown.min() * 100

    print(f"💰 累積淨利（含手續費）: ${total_profit:,.0f}")
    print(f"📈 總報酬率            : {total_profit_pct:.2f}%")
    print(f"🏆 勝率                : {win_rate:.2f}%")
    print(f"────────────────────────────────")
    print(f"📐 策略夏普比率（年化）: {sharpe:.4f}")
    print(f"📐 持有夏普比率（年化）: {hold_sharpe:.4f}  ← 同期單純持有")
    print(f"────────────────────────────────")
    print(f"📉 策略最大回撤        : {max_drawdown:.2f}%")
    print(f"📉 持有最大回撤        : {hold_max_drawdown:.2f}%  ← 同期單純持有")
    print(f"────────────────────────────────")
    print(f"📈 單純持有報酬率      : {hold_return_pct:.2f}%  ← 同期單純持有")

    # ==========================================
    # 💾 5. 匯出 JSON / CSV
    # ==========================================
    kpi_data = {
        "stock_id"          : STOCK_ID,
        "max_capital"       : float(TOTAL_CAPITAL),
        "total_profit"      : round(float(total_profit), 2),
        "total_return_pct"  : round(total_profit_pct, 2),
        "win_rate"          : round(win_rate, 2),
        "total_trades"      : len(sell_trades),
        "sharpe_ratio"      : round(sharpe, 4),
        "hold_sharpe_ratio" : round(hold_sharpe, 4),
        "hold_return_pct"   : round(hold_return_pct, 2),
        "max_drawdown_pct"  : round(max_drawdown, 2),
        "hold_max_drawdown_pct": round(hold_max_drawdown, 2), # 🌟 新增這裡！
        "current_status"    : "HOLDING" if holding_qty > 0 else "EMPTY"
    }

    export_data = {
        "kpi"   : kpi_data,
        "trades": trade_df.to_dict(orient='records') if not trade_df.empty else []
    }

    with open(f"web_data_{STOCK_ID}_daily.json", 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=4)

    df_kline = df_res[['date', 'open', 'high', 'low', 'close', 'SMA_20']].copy()
    df_kline['date'] = df_kline['date'].dt.strftime('%Y-%m-%d')
    df_kline.to_csv(f"web_kline_{STOCK_ID}_daily.csv", index=False)

 # ==========================================
    # 📱 [新增] 6. LINE 每日最新訊號過濾與發送
    # ==========================================
    # 鎖定迴圈最後一天的日期與價格，確保絕不重複發送舊歷史訊號
    last_date = df_res['date'].iloc[-1].strftime('%Y-%m-%d')
    last_close_price = df_res['close'].iloc[-1]

    # 尋找「最後一天」是否有觸發 BUY 或 SELL
    today_action = None
    for t in trades:
        if t['date'] == last_date and t['action'] in ['BUY', 'SELL']:
            today_action = t['action']
            break

    if today_action:
        # daily_test.py 裡面的股票皆為 AI 核心股 (is_ai_stock=True)
        line_msg = generate_trade_signal_msg(
            date=last_date,
            stock_id=STOCK_ID,
            action=today_action,
            price=last_close_price,
            is_ai_stock=False 
        )
        send_line_message(line_msg)
        print(f"今日 ({last_date}) {STOCK_ID} 觸發 {today_action} 訊號，已發送 LINE 通知。")
    else:
        print(f"今日 ({last_date}) {STOCK_ID} 無買賣訊號，維持現有狀態，不發送通知。")

print("\n所有標的每日跟進完畢含手續費、夏普比率、同期持有最大回撤比較、LINE推播判斷")