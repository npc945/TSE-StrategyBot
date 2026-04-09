import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
import google.generativeai as genai
# ==========================================
# ⚙️ 1. 頁面基本設定
# ==========================================
st.set_page_config(page_title="AI 量化交易回測系統", page_icon="📈", layout="wide")

STOCK_DICT = {
    "2330": "台積電 (2330)",
    "2454": "聯發科 (2454)",
    "2317": "鴻海 (2317)"
}

st.sidebar.title("策略參數與展示")
selected_id = st.sidebar.radio("🎯 選擇股票標的", list(STOCK_DICT.keys()), format_func=lambda x: f"{x} {STOCK_DICT[x]}")

# 2. 讀取對應資料 (同時讀取 Proof 與 Daily)
proof_json_file = f"web_data_{selected_id}_proof.json"
proof_csv_file = f"web_kline_{selected_id}.csv"

daily_json_file = f"web_data_{selected_id}_daily.json"
daily_csv_file = f"web_kline_{selected_id}_daily.csv"

if not os.path.exists(proof_json_file) or not os.path.exists(daily_json_file):
    st.error(f"找不到 {selected_id} 的完整資料！請確認 backtest_proof.py 與 daily_test.py 皆已執行。")
    st.stop()

# 讀取 Proof (靜態回測)
with open(proof_json_file, 'r', encoding='utf-8') as f:
    proof_data = json.load(f)
kpi_proof = proof_data['kpi']
df_trades_proof = pd.DataFrame(proof_data['trades'])
df_kline_proof = pd.read_csv(proof_csv_file)
df_kline_proof['date'] = pd.to_datetime(df_kline_proof['date'])
if not df_trades_proof.empty:
    df_trades_proof['date'] = pd.to_datetime(df_trades_proof['date'])

# 讀取 Daily (每日實盤)
with open(daily_json_file, 'r', encoding='utf-8') as f:
    daily_data = json.load(f)
kpi_daily = daily_data['kpi']
df_trades_daily = pd.DataFrame(daily_data['trades'])
df_kline_daily = pd.read_csv(daily_csv_file)
df_kline_daily['date'] = pd.to_datetime(df_kline_daily['date'])
if not df_trades_daily.empty:
    df_trades_daily['date'] = pd.to_datetime(df_trades_daily['date'])
st.sidebar.markdown("---")
st.sidebar.subheader("💰 互動資金模擬器")
st.sidebar.markdown("拖曳下方滑桿，即時模擬不同本金的獲利變化：")
# 讓教授可以從 10萬 拉到 500萬
simulated_capital = st.sidebar.slider("模擬投入本金 (萬台幣)", min_value=10, max_value=500, value=100, step=10) * 10000

# 計算模擬後的絕對獲利 (用真實的 % 數去乘上教授拉的本金)
simulated_profit_proof = (kpi_proof['total_return_pct'] / 100) * simulated_capital
simulated_profit_daily = (kpi_daily['total_return_pct'] / 100) * simulated_capital

st.sidebar.success(f"若投入 {simulated_capital:,.0f} 元：\n\n回測預期淨利將達 **${simulated_profit_proof:,.0f}**")
# 頂部狀態列 (今日即時戰報)
st.title(f"{selected_id} {STOCK_DICT[selected_id]} - AI 量化系統展示")

current_status = kpi_daily.get('current_status', 'EMPTY')
if current_status == "HOLDING":
    st.success(f"【今日實盤狀態】系統目前正在持有 {STOCK_DICT[selected_id]}，即時追蹤未實現損益中！")
else:
    st.info(f"【今日實盤狀態】系統目前對 {STOCK_DICT[selected_id]} 保持空手觀望，等待下一次策略發送訊號。")

st.markdown("本系統採用雙層架構：底層由 AI 預測波段起漲點，表層套用**「固定資金控管與量化濾網」**，實踐大賺小賠之穩定獲利模型。")

# 畫圖函數
def create_kline_chart(df_k, df_t):
    fig = go.Figure()
    # K 線圖
    fig.add_trace(go.Candlestick(
        x=df_k['date'], open=df_k['open'], high=df_k['high'], low=df_k['low'], close=df_k['close'],
        name='K線', increasing_line_color='#ff4b4b', decreasing_line_color='#00cc96',
    ))
    # 20日均線
    fig.add_trace(go.Scatter(
        x=df_k['date'], y=df_k['SMA_20'], mode='lines', line=dict(color='white', width=1.5), name='20日均線'
    ))

    # 買賣點標記
    if not df_t.empty:
        buy_points = df_t[df_t['action'] == 'BUY']
        sell_points = df_t[df_t['action'].str.contains('SELL')]
        hold_points = df_t[df_t['action'].str.contains('HOLDING')] 

        if not buy_points.empty:
            fig.add_trace(go.Scatter(x=buy_points['date'], y=buy_points['price'] * 0.96, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00ff00', line=dict(width=1, color='black')), name='🚀 買進'))
        if not sell_points.empty:
            fig.add_trace(go.Scatter(x=sell_points['date'], y=sell_points['price'] * 1.04, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff0000', line=dict(width=1, color='black')), name='📉 賣出'))
        if not hold_points.empty:
            fig.add_trace(go.Scatter(x=hold_points['date'], y=hold_points['price'], mode='markers', marker=dict(symbol='star', size=16, color='gold', line=dict(width=1, color='black')), name='⭐ 今日持倉結算'))

    fig.update_layout(
        dragmode='pan', xaxis=dict(rangeslider=dict(visible=False), type="date"), yaxis=dict(autorange=True, fixedrange=False),
        height=500, margin=dict(l=10, r=10, t=10, b=10), hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    return fig

# 表格上色邏輯
def color_profit(val):
    if isinstance(val, (int, float)):
        color = '#ff4b4b' if val > 0 else '#00cc96' if val < 0 else 'white'
        return f'color: {color}; font-weight: bold;'
    return ''

# 表格數字格式化
format_dict = {
    'qty': '{:,.0f} 股',
    'price': '${:,.2f}',
    'profit': '${:,.0f}',
    'profit_pct': '{:.2f}%'
}

# 分頁設定
tab1, tab2, tab3 = st.tabs(["第一部分：AI 預測能力", "第二部分：AI+策略的回測績效證明", "第三部分：每日實盤跟進"])

# Tab 1：AI預測力分析
with tab1:
    st.header("AI 原始預測能力驗證")
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
        st.header("AI核心預測指標")
        st.markdown("這裡展示 AI 在測試集上(未見過的資料)的原始預測能力。**預測目標為：未來20天內漲幅是否能達到4%**。此處未套用任何量化濾網，純粹檢驗 AI 核心大腦的勝率。")
        st.metric("測試集總天數", f"{current_stat['total']} 天")
        st.metric("判定具備潛力", f"{current_stat['predicted_up']} 次", delta=f"觸發率 {signal_rate:.1f}%")
        st.metric("訊號精確率 (Precision)", f"{current_stat['precision']}%")
    with col_b:
        fig_funnel = go.Figure(go.Funnel(y=["測試集總天數", "AI 判定潛力", "實際達成 4% 目標"], x=[current_stat["total"], current_stat["predicted_up"], actual_success], textinfo="value+percent initial", marker={"color": ["#1f77b4", "#ff7f0e", "#2ca02c"]}))
        fig_funnel.update_layout(margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_funnel, use_container_width=True)

# Tab 2：實戰回測績效證明
with tab2:
    st.header("策略實戰績效 (學術回測驗證)")
    st.markdown("本頁面展示了 AI 結合技術分析濾網的回測績效。為確保績效公允並排除股價通膨干擾，**本回測採用「單檔固定本金 100 萬與零股交易」進行運算**，真實呈現策略的極致盈虧比。")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總報酬率", f"{kpi_proof['total_return_pct']}%")
    col2.metric("累積淨利", f"${kpi_proof['total_profit']:,.0f}")
    col3.metric("策略勝率", f"{kpi_proof['win_rate']}%", f"共 {kpi_proof['total_trades']} 次交易")
    col4.metric("單檔固定本金", f"${kpi_proof['max_capital']:,.0f}") # 🌟 修正名稱

    st.plotly_chart(create_kline_chart(df_kline_proof, df_trades_proof), use_container_width=True, config={'scrollZoom': True})

    if not df_trades_proof.empty:
        df_show_proof = df_trades_proof[['date', 'action', 'qty', 'price', 'profit', 'profit_pct']].copy()
        df_show_proof['date'] = df_show_proof['date'].dt.strftime('%Y-%m-%d')
        # 🌟 加上千分位與小數點格式化
        st.dataframe(df_show_proof.style.map(color_profit, subset=['profit', 'profit_pct']).format(format_dict), use_container_width=True, height=300)

# Tab 3：每日實盤跟進
with tab3:
    st.header("每日實盤跟進")
    st.markdown("本頁面串接每日自動化更新系統，展示從測試集起點一路上線至今的真實動態績效。**系統嚴格執行固定資金控管（最高投入100萬），並即時結算未實現損益。**")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("實盤總報酬率", f"{kpi_daily['total_return_pct']}%")
    col2.metric("實盤累積淨利", f"${kpi_daily['total_profit']:,.0f}")
    col3.metric("實盤勝率", f"{kpi_daily['win_rate']}%", f"共 {kpi_daily['total_trades']} 次交易")
    col4.metric("單檔固定本金", f"${kpi_daily['max_capital']:,.0f}") # 🌟 修正名稱

    st.plotly_chart(create_kline_chart(df_kline_daily, df_trades_daily), use_container_width=True, config={'scrollZoom': True})

    if not df_trades_daily.empty:
        df_show_daily = df_trades_daily[['date', 'action', 'qty', 'price', 'profit', 'profit_pct']].copy()
        df_show_daily['date'] = df_show_daily['date'].dt.strftime('%Y-%m-%d')
        # 🌟 加上千分位與小數點格式化
        st.dataframe(df_show_daily.style.map(color_profit, subset=['profit', 'profit_pct']).format(format_dict), use_container_width=True, height=300)

st.markdown("---")
st.header("🤖 系統專屬 LLM 量化助理")

# 1. 確保對話紀錄存在
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 把過去的對話畫在畫面上
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 3. 接收使用者輸入
if prompt := st.chat_input("請輸入您的指令或問題，例如：『目前戰報如何？』"):
    
    # 顯示使用者的訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 助理開始回應
    with st.chat_message("assistant"):
        with st.spinner("🧠 AI 正在解析系統數據..."):
            try:
                # 讀取環境變數中的金鑰 (標準化命名)
                # 請確保你的 .env 檔案裡面寫的是 GEMINI_API_KEY=你的新金鑰
                api_key = os.getenv('gemini_api') 
                
                if not api_key:
                    st.error("⚠️ 系統提示：找不到 API 金鑰。請確認 .env 檔案中已設定 GEMINI_API_KEY。")
                else:
                    # 初始化 API
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    # 🌟 完整專業版的 Context (將系統的動態數據無縫餵給 LLM)
                    system_context = f"""
                    你現在是這套「AI 量化交易系統」的專屬解說員。
                    目前使用者正在查看的股票是：{STOCK_DICT.get(selected_id, '未知')}。
                    這檔股票的目前系統狀態是：【{current_status}】。
                    歷史回測總報酬率為：{kpi_proof.get('total_return_pct', 0)}%，勝率為：{kpi_proof.get('win_rate', 0)}%。
                    每日實盤跟進累積淨利為：{kpi_daily.get('total_profit', 0)} 元。
                    
                    請根據上述真實系統數據，以專業、精煉且充滿自信的金融分析師語氣回答使用者的問題。
                    絕對不要憑空捏造上述沒有提供的績效數據。
                    使用者的問題是：{prompt}
                    """
                    
                    # 發送請求
                    response = model.generate_content(system_context)
                    
                    # 顯示並儲存成功的回應
                    st.write(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    
            except Exception as e:
                # 正式環境的專業錯誤提示
                st.error(f"⚠️ API 呼叫發生錯誤，請稍後再試或檢查網路連線。錯誤細節：{str(e)}")