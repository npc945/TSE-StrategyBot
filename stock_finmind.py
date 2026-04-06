from FinMind.data import DataLoader
import os
from dotenv import load_dotenv
import pandas as pd
import datetime
from sqlalchemy import create_engine
from sql_upsert import upsert
# columns=(
# ['date', 'stock_id', 'Trading_Volume', 'Trading_money',
# 'open', 'max','min', 'close', 'spread', 'Trading_turnover']
# 2. 獲取這支 Python 檔案所在的「絕對路徑資料夾」
current_dir = os.path.dirname(os.path.abspath(__file__))
# 3. 將資料夾路徑與 token.env 檔名結合
env_path = os.path.join(current_dir, "token.env")

# 4. 讀取絕對路徑下的 token.env
load_dotenv(env_path)
token=os.getenv('token')
sql_engine=os.getenv("sql_engine")
#連線mysql
engine=create_engine(sql_engine)

today=datetime.date.today().strftime("%Y-%m-%d")#轉字串
api=DataLoader()
api.login_by_token(token)

stock_list=["2330","2317","2603","2454","2881"]
start_date="2015-01-01"

def stock_data(id,start,end):
    return api.taiwan_stock_daily(id,start,end)

data = pd.read_sql("select stock_id, date from stock_data", engine)
latest_date = data.groupby("stock_id")["date"].max().to_dict()#找個股最後取資料日期

all_data=[]
for id in stock_list:
    if id in latest_date:
        # 從最新日期的下一天開始抓
        start_date = (pd.to_datetime(latest_date[id]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = "2015-01-01"
    if start_date > today:
        continue
    all_data.append(stock_data(id, start_date, today))
all_data=pd.concat(all_data)
all_data=all_data.sort_values(by=["stock_id","date"])
#因為要計算kd所以把max改high, min改low
all_data.rename(columns={"max":"high","min":"low"},inplace=True)#inplace=true針對原資料修改
all_data = all_data[all_data['Trading_Volume'] != 0]
#由sql_upsert.py去寫入db
upsert(engine,
       "stock_data",
       all_data,
       update_cols=["Trading_Volume", "Trading_money", "open", "high", "low", "close", "spread", "Trading_turnover"]
       )
