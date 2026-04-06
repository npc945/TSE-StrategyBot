import pandas as pd
import numpy as np
import pandas_ta as ta
from sqlalchemy import create_engine
from sql_upsert import upsert
import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "token.env")
load_dotenv(env_path)
sql_engine=os.getenv('sql_engine')#讀取環境變數token
# 連線到資料庫
engine = create_engine(sql_engine)
data = pd.read_sql("select * from stock_data", engine)

stock_id = data['stock_id'].unique()
tech = []

for id in stock_id:
    df = data[data['stock_id'] == id].copy()
    df = df.sort_values(by="date")
    
    # 計算5日與20日移動平均線
    df['SMA_5'] = df.ta.sma(length=5)
    df['SMA_10'] = df.ta.sma(length=10)
    df['SMA_20'] = df.ta.sma(length=20)
    
    # --- 計算乖離率 (Bias) ---
    df['Bias_20'] = (df['close'] - df['SMA_20']) / df['SMA_20']

    # 計算14日RSI
    df['RSI_14'] = df.ta.rsi(length=14)
    
    # 計算MACD
    macd = df.ta.macd() #dataframe的結果
    df['MACD'] = macd['MACD_12_26_9'] #快線
    df['MACD_signal'] = macd['MACDs_12_26_9'] #慢線
    df['MACD_diff'] = macd['MACDh_12_26_9'] #差離值
    
    # 計算kd
    kd = df.ta.stoch(k=9, d=3, smooth_k=3) 
    df['K'] = kd['STOCHk_9_3_3'] #k值
    df['D'] = kd['STOCHd_9_3_3'] #d值
    
    #新增計算ADX平均趨向指標
    adx = df.ta.adx(length=14)
    df['ADX_14'] = adx['ADX_14'] 
    
    tech.append(df)

data = pd.concat(tech)
data = data.sort_values(by=["stock_id", "date"])

# 資料表改空列表nan改none sql才能懂
data = data.replace({np.nan: None})

upsert(engine,
       "stock_data",
       data,
       # 🌟 [修改] 將 ADX_14 加進 update_cols 的更新清單中
       update_cols=["SMA_5", "SMA_20", "SMA_10", "RSI_14", "MACD", "MACD_signal", "MACD_diff", "K", "D", "Bias_20", "ADX_14"]
)