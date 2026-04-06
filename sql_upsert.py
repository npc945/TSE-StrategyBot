import pandas as pd
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.mysql import insert

def upsert(engine, table_name, df, update_cols):
    # 如果df沒資料就直接結束
    if df is None or df.empty:
        return 0

    #轉字典
    rows = df.to_dict(orient="records")
    if len(rows) == 0:
        return 0

    #取得資料表結構
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)

    # 建立 insert 指令
    stmt = insert(table).values(rows)

    #主鍵重複時只更新 update_cols
    update_dict = {}
    for c in update_cols:
        update_dict[c] = stmt.inserted[c]
    stmt = stmt.on_duplicate_key_update(**update_dict)

    #寫入
    with engine.begin() as conn:
        res = conn.execute(stmt)
        return res.rowcount or 0