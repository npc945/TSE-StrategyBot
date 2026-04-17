import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "token.env")
load_dotenv(env_path)

st.set_page_config(page_title="量化交易回測系統", page_icon="📈", layout="wide")

STOCK_DICT = {
    "2330": "台積電 (2330)",
    "2454": "聯發科 (2454)",
    "2317": "鴻海 (2317)",
    "2382": "廣達 (2382)",
    "2308": "台達電 (2308)",
    "2881": "富邦金 (2881)",
    "2603": "長榮 (2603)",
    "1301": "台塑 (1301)",
    "1513": "中興電 (1513)",
    "2412": "中華電 (2412)"
}

AI_STOCKS = ["2330", "2454", "2317"]

st.sidebar.title("策略參數與展示")

st.sidebar.markdown("標的選擇：根據是否加入 AI 預測模型，分為兩個群組。")
stock_group = st.sidebar.radio(
    "1. 選擇策略群組：", 
    ["有加入AI策略(3檔)", "尚未加入AI策略(7檔)"]
)

# 根據選擇的群組，過濾要顯示的股票清單
if "有" in stock_group:
    current_options = {k: v for k, v in STOCK_DICT.items() if k in AI_STOCKS}
else:
    current_options = {k: v for k, v in STOCK_DICT.items() if k not in AI_STOCKS}

selected_id = st.sidebar.radio(
    "2. 選擇分析標的：", 
    list(current_options.keys()), 
    format_func=lambda x: f"{x} {current_options[x]}"
)

is_ai_stock = selected_id in AI_STOCKS

if is_ai_stock:
    proof_json_file = f"web_data_{selected_id}_proof.json"
    proof_csv_file = f"web_kline_{selected_id}.csv"
    daily_json_file = f"web_data_{selected_id}_daily.json"
    daily_csv_file = f"web_kline_{selected_id}_daily.csv"

    if not os.path.exists(proof_json_file) or not os.path.exists(daily_json_file):
        st.error(f"找不到 {selected_id} 的完整資料")
        st.stop()

    with open(proof_json_file, 'r', encoding='utf-8') as f:
        proof_data = json.load(f)
    kpi_proof = proof_data['kpi']
    df_trades_proof = pd.DataFrame(proof_data['trades'])
    df_kline_proof = pd.read_csv(proof_csv_file)
    df_kline_proof['date'] = pd.to_datetime(df_kline_proof['date'])
    if not df_trades_proof.empty: df_trades_proof['date'] = pd.to_datetime(df_trades_proof['date'])

else:
    daily_json_file = f"web_data_{selected_id}_daily.json"
    daily_csv_file = f"web_kline_{selected_id}_daily.csv"
    
    if not os.path.exists(daily_json_file):
        st.error(f"找不到 {selected_id} 的實盤資料")
        st.stop()
        
    kpi_proof = {}

with open(daily_json_file, 'r', encoding='utf-8') as f:
    daily_data = json.load(f)
kpi_daily = daily_data['kpi']
df_trades_daily = pd.DataFrame(daily_data['trades'])
df_kline_daily = pd.read_csv(daily_csv_file)
df_kline_daily['date'] = pd.to_datetime(df_kline_daily['date'])
if not df_trades_daily.empty: df_trades_daily['date'] = pd.to_datetime(df_trades_daily['date'])

#側邊欄位
st.sidebar.markdown("---")
st.sidebar.subheader("資金規模模擬器")
st.sidebar.markdown("調整下方本金參數，即時估算策略的預期損益：")
simulated_capital = st.sidebar.slider("模擬投入本金 (萬台幣)", min_value=10, max_value=500, value=100, step=10) * 10000

simulated_profit_daily = (kpi_daily.get('total_return_pct', 0) / 100) * simulated_capital
st.sidebar.success(f"若固定投入 {simulated_capital:,.0f} 元：\n\n系統預期淨利為 **${simulated_profit_daily:,.0f}**")

st.title(f"{selected_id} {STOCK_DICT[selected_id]} - 量化策略分析儀表板")

current_status = kpi_daily.get('current_status', 'EMPTY')
if current_status == "HOLDING":
    st.success(f"【即時倉位狀態】系統目前持有 {STOCK_DICT[selected_id]}，持續動態追蹤未實現損益。")
else:
    st.info(f"【即時倉位狀態】系統目前對 {STOCK_DICT[selected_id]} 保持空手觀望，等待技術訊號觸發。")

if is_ai_stock:
    st.markdown("本標的採用雙層架構：結合 LSTM 模型預測波段趨勢，表層套用技術濾網與資金控管，實踐風險調整後之穩定報酬。")
else:
    st.markdown("本標的單純採用技術分析策略：透過技術指標之複合條件進行交易，作為本系統評估風險的基本對照。")

st.caption("系統備註：本展示之歷史回測已扣除千分之 1.425 券商手續費與千分之 3 證交稅，呈現真實摩擦成本下之淨利。")

# 視覺化圖表與 UI 模組函數
def create_kline_chart(df_k, df_t):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_k['date'], open=df_k['open'], high=df_k['high'], low=df_k['low'], close=df_k['close'],
        name='K線', increasing_line_color='#ff4b4b', decreasing_line_color='#00cc96',
    ))
    fig.add_trace(go.Scatter(
        x=df_k['date'], y=df_k['SMA_20'], mode='lines', line=dict(color='white', width=1.5), name='20日均線'
    ))

    if not df_t.empty:
        buy_points = df_t[df_t['action'] == 'BUY']
        sell_points = df_t[df_t['action'].str.contains('SELL')]
        hold_points = df_t[df_t['action'].str.contains('HOLDING')] 

        if not buy_points.empty:
            fig.add_trace(go.Scatter(x=buy_points['date'], y=buy_points['price'] * 0.96, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00ff00', line=dict(width=1, color='black')), name=' 買進'))
        if not sell_points.empty:
            fig.add_trace(go.Scatter(x=sell_points['date'], y=sell_points['price'] * 1.04, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff0000', line=dict(width=1, color='black')), name=' 賣出'))
        if not hold_points.empty:
            fig.add_trace(go.Scatter(x=hold_points['date'], y=hold_points['price'], mode='markers', marker=dict(symbol='star', size=16, color='gold', line=dict(width=1, color='black')), name='⭐ 今日結算'))

    fig.update_layout(
        dragmode='pan', xaxis=dict(rangeslider=dict(visible=False), type="date"), yaxis=dict(autorange=True, fixedrange=False),
        height=500, margin=dict(l=10, r=10, t=10, b=10), hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    return fig

def color_profit(val):
    if isinstance(val, (int, float)):
        color = '#ff4b4b' if val > 0 else '#00cc96' if val < 0 else 'white'
        return f'color: {color}; font-weight: bold;'
    return ''

format_dict = {'qty': '{:,.0f} 股', 'price': '${:,.2f}', 'actual_cost': '${:,.2f}', 'profit': '${:,.0f}', 'profit_pct': '{:.2f}%'}

# 數據儀表板
def render_performance_dashboard(kpi_data, df_k, df_t):
    # 第一層：獲利能力
    st.subheader("絕對獲利指標")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("策略總報酬率", f"{kpi_data.get('total_return_pct', 0)}%", delta=f"同期單純持有 {kpi_data.get('hold_return_pct', 0)}%", delta_color="off")
    c2.metric("累積淨利 (已扣手續費)", f"${kpi_data.get('total_profit', 0):,.0f}")
    c3.metric("策略勝率", f"{kpi_data.get('win_rate', 0)}%", f"交易總數 {kpi_data.get('total_trades', 0)} 次")
    c4.metric("單檔固定本金", f"${kpi_data.get('max_capital', 1000000):,.0f}")

    # 風險控制
    st.subheader("風控與波動指標")
    cA, cB, cC, cD = st.columns(4)
    cA.metric("策略夏普比率", f"{kpi_data.get('sharpe_ratio', 0):.4f}")
    cB.metric("單純持有夏普比率", f"{kpi_data.get('hold_sharpe_ratio', 0):.4f}")
    
    # 計算 MDD 避開多少跌幅
    mdd = kpi_data.get('max_drawdown_pct', 0)
    hold_mdd = kpi_data.get('hold_max_drawdown_pct', 0)
    mdd_saved = abs(hold_mdd) - abs(mdd)
    delta_text = f"減少 {mdd_saved:.2f}% 資產回撤" if mdd_saved > 0 else f"落後 {abs(mdd_saved):.2f}%"
    
    cC.metric("策略最大回撤", f"{mdd:.2f}%", delta=delta_text, delta_color="normal")
    cD.metric("單純持有最大回撤", f"{hold_mdd:.2f}%")

    st.plotly_chart(create_kline_chart(df_k, df_t), use_container_width=True, config={'scrollZoom': True})

    if not df_t.empty:
        #actual_cost交易成本
        df_show = df_t[['date', 'action', 'qty', 'price', 'actual_cost', 'profit', 'profit_pct']].copy()
        df_show['date'] = df_show['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(df_show.style.map(color_profit, subset=['profit', 'profit_pct']).format(format_dict), use_container_width=True, height=300)

if is_ai_stock:
    tab1, tab2, tab3 = st.tabs(["第一部分：LSTM 模型預測驗證", "第二部分：策略回測績效", "第三部分：實盤動態追蹤"])

    with tab1:
        st.header("LSTM 模型預測精度分析")
        ai_stats = {
            "2317": {"total": 498, "predicted_up": 95, "precision": 64.21},
            "2330": {"total": 499, "predicted_up": 206, "precision": 68.93},
            "2454": {"total": 499, "predicted_up": 146, "precision": 75.34}
        }
        current_stat = ai_stats.get(selected_id, {"total": 500, "predicted_up": 100, "precision": 50.0})
        actual_success = int(current_stat["predicted_up"] * (current_stat["precision"] / 100))
        signal_rate = (current_stat["predicted_up"] / current_stat["total"]) * 100

        col_a, col_b = st.columns([1, 1.5])
        with col_a:
            st.markdown("展示 LSTM 在測試集(測試集代表模型沒見過的數據)的原始預測表現。預測目標：未來 20 個交易日內漲幅達 4% 的機率。此處尚未使用資金，單純評估模型辨識波段的精準度。")
            st.metric("測試集總天數", f"{current_stat['total']} 天")
            st.metric("模型觸發訊號", f"{current_stat['predicted_up']} 次", delta=f"發動頻率 {signal_rate:.1f}%")
            st.metric("預測精準度 (Precision)", f"{current_stat['precision']}%")
        with col_b:
            fig_funnel = go.Figure(go.Funnel(y=["測試集總天數", "模型判定具備潛力", "實際達成波段目標"], x=[current_stat["total"], current_stat["predicted_up"], actual_success], textinfo="value+percent initial", marker={"color": ["#1f77b4", "#ff7f0e", "#2ca02c"]}))
            fig_funnel.update_layout(margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_funnel, use_container_width=True)

    with tab2:
        st.header("策略回測績效")
        st.markdown("展示AI模型結合技術面後的回測結果。著重於最大回撤 (MDD) 控制與風險調整後報酬 (Sharpe Ratio) 之驗證。")
        render_performance_dashboard(kpi_proof, df_kline_proof, df_trades_proof)

    with tab3:
        st.header("實盤動態追蹤")
        st.markdown("串接系統每日自動化更新真實數據，嚴格執行資金控管上限，並即時結算未實現損益與摩擦成本。")
        render_performance_dashboard(kpi_daily, df_kline_daily, df_trades_daily)

else:
    tab1 = st.tabs(["量化策略實盤追蹤 (未結合AI的部分)"])[0]
    with tab1:
        st.header("純技術分析量化實盤")
        st.markdown("純量化技術分析策略，未疊加 AI 預測模型。主要作為系統擴充選股池的基礎策略，透過自動化的技術指標條件進出，提供穩定且輕量化的程式交易選項。")
        render_performance_dashboard(kpi_daily, df_kline_daily, df_trades_daily)


st.markdown("---")
st.header("系統專屬量化分析助理")


if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("請輸入您選擇的股票進行詢問，例如：請幫我解讀此檔股票目前的夏普比率與最大回撤。"):
    
    # 🔒 [新增] 檢查鎖的狀態：如果還在跑，就擋下來！
    if st.session_state.is_processing:
        st.warning("⚠️ AI 正在努力分析中，請勿重複送出問題！")
    else:
        # 🔒 [新增] 上鎖：宣告系統開始執行
        st.session_state.is_processing = True

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在整合系統歷史回測與風險評估數據..."):
                try:

                    api_key = os.getenv('gemini_api') 
                    if not api_key:
                        try:
                            api_key = st.secrets["gemini_api"]
                        except:
                            api_key = None

                    if not api_key:
                        st.error("no api key")
                    else:
                        genai.configure(api_key=api_key)
                        # 指定模型
                        model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')
                        
                        group_desc = "配備 LSTM 預測模型的 AI 核心標的" if is_ai_stock else "未配備 AI 之純技術分析對照組標的"
                        
                        # LLM
                        system_context = f"""
                        你現在是這套「量化交易系統」的專屬數據分析師。用語必須客觀、專業，著重於風險控管與摩擦成本的分析。
                        目前使用者正在查看的股票是：{STOCK_DICT.get(selected_id, '未知')} ({group_desc})。
                        
                        【當前系統核心數據】
                        1. 實盤狀態：{current_status} (HOLDING 代表持倉中，EMPTY 代表空手)
                        2. 策略總報酬率：{kpi_daily.get('total_return_pct', 0)}% (同期單純持有：{kpi_daily.get('hold_return_pct', 0)}%)
                        3. 累積淨利(已扣手續費)：{kpi_daily.get('total_profit', 0)} 元
                        4. 策略勝率：{kpi_daily.get('win_rate', 0)}%
                        5. 策略夏普比率：{kpi_daily.get('sharpe_ratio', 0)} (同期單純持有夏普：{kpi_daily.get('hold_sharpe_ratio', 0)})
                        6. 策略最大回撤(MDD)：{kpi_daily.get('max_drawdown_pct', 0)}% (同期單純持有MDD：{kpi_daily.get('hold_max_drawdown_pct', 0)}%)
                        
                        請根據上述真實數據回答使用者問題。嚴禁捏造數據。特別注意如果策略的 MDD 絕對值小於大盤，代表策略成功展現了避險與抗跌能力。
                        使用者的問題是：{prompt}
                        """
                        
                        response = model.generate_content(system_context)
                        st.write(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                
                
                except Exception as e:
                    err = str(e)
                    if "429" in err or "Quota" in err:
                        st.warning("⚠️ 流量達到上限（每分鐘 5 次）。請稍候 1 分鐘後再發問！")
                    else:
                        st.error(f"連線異常，錯誤細節：{err}")
                
                finally:
                    st.session_state.is_processing = False