import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import os
from dotenv import load_dotenv
import joblib
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "token.env")
load_dotenv(env_path)
sql_engine=os.getenv('sql_engine')#讀取環境變數token

CONN_STR   = sql_engine
TABLE      = "stock_data"
STOCK_ID   = 2330
LOOK_BACK  = 20
EPOCHS     = 50
BATCH_SIZE = 32
N_SPLITS   = 5

selected_features = ["Trading_Volume","Bias_20"]#"MACD_diff","Trading_Volume",

# 資料前處理
engine = create_engine(CONN_STR)
cols_sql = ', '.join(selected_features)
sql = f"SELECT date, close, {cols_sql} FROM {TABLE} WHERE stock_id={STOCK_ID}"
df = pd.read_sql(sql, engine)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 🌟 修正特徵滯後：為了避免未來函數，所有特徵往後推一天
for c in selected_features:
    df[c] = df[c].shift(1)

# 抓出「從明天開始算，未來 10 天內的最高價」
df['future_max_5d'] = df['close'].rolling(window=20).max().shift(-20)

# 考卷答案：未來 5 天內的高點，只要大於今天收盤價的 2%，這筆交易就值得做 (標記為 1)
df['target'] = (df['future_max_5d'] > df['close'] * 1.04).astype(int)

# 砍掉有 NaN 的資料 (最前面因為 shift(1) 會有 1 筆，最後面因為 shift(-5) 會有 5 筆)
df = df.dropna().reset_index(drop=True)

test_split_idx = int(len(df) * 0.8)
df_dev = df.iloc[:test_split_idx].copy()  # 開發集 (用來做 Cross-Validation)
df_test = df.iloc[test_split_idx:].copy() # 測試集 (最終考卷)
print(f"開發集樣本數 (Development Set): {len(df_dev)}")
print(f"測試集樣本數 (Hold-out Test):   {len(df_test)}")

# 核心函數定義
def create_dataset(X, y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)])
        ys.append(y[i + look_back])
    return np.array(Xs), np.array(ys)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(16, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

print("\n🔄 執行 Walk-Forward Cross Validation (5-Fold)...")

X_dev_raw = df_dev[selected_features].values
y_dev_raw = df_dev["target"].values

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
cv_scores = []

for fold, (train_index, val_index) in enumerate(tscv.split(X_dev_raw)):
    X_tr, X_val = X_dev_raw[train_index], X_dev_raw[val_index]
    y_tr, y_val = y_dev_raw[train_index], y_dev_raw[val_index]
    
    scaler = MinMaxScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    
    X_tr_3d, y_tr_3d = create_dataset(X_tr_s, y_tr, LOOK_BACK)
    X_val_3d, y_val_3d = create_dataset(X_val_s, y_val, LOOK_BACK)
    
    model = build_model((X_tr_3d.shape[1], X_tr_3d.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_tr_3d, y_tr_3d, epochs=EPOCHS, batch_size=BATCH_SIZE, 
              validation_data=(X_val_3d, y_val_3d), callbacks=[early_stop], verbose=0)
    
    pred_prob = model.predict(X_val_3d, verbose=0)
    pred_class = (pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_val_3d, pred_class)
    cv_scores.append(acc)
    
    print(f"   Fold {fold+1}: Accuracy = {acc:.4f}")

print(f"🎯 平均驗證準確率 (Average CV Accuracy): {np.mean(cv_scores):.4f}")


print("\n🚀 [Step 3] 執行最終模型訓練 (Retrain on Full Dev Set)...")

scaler_final = MinMaxScaler()
X_dev_s = scaler_final.fit_transform(X_dev_raw)
y_dev = y_dev_raw

X_test_raw = df_test[selected_features].values
y_test = df_test["target"].values
X_test_s = scaler_final.transform(X_test_raw)

X_dev_3d, y_dev_3d = create_dataset(X_dev_s, y_dev, LOOK_BACK)
X_test_3d, y_test_3d = create_dataset(X_test_s, y_test, LOOK_BACK)

split_val = int(len(X_dev_3d) * 0.9)
X_train_final = X_dev_3d[:split_val]
y_train_final = y_dev_3d[:split_val]
X_val_final = X_dev_3d[split_val:]
y_val_final = y_dev_3d[split_val:]

final_model = build_model((X_train_final.shape[1], X_train_final.shape[2]))
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

final_model.fit(X_train_final, y_train_final, epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_data=(X_val_final, y_val_final),
                callbacks=[early_stop], verbose=1)

print("\n📊 [Step 4] 最終測試結果 (Hold-out Test Evaluation)")
y_pred_prob = final_model.predict(X_test_3d, verbose=0)
y_pred_class = (y_pred_prob > 0.5).astype(int)

final_acc = accuracy_score(y_test_3d, y_pred_class)
final_prec = precision_score(y_test_3d, y_pred_class, zero_division=0)
conf_matrix = confusion_matrix(y_test_3d, y_pred_class)

print("="*40)
print(f"方向準確率 (Accuracy) : {final_acc:.4f}")
print(f"訊號精確率 (Precision): {final_prec:.4f}")
print("="*40)
print("混淆矩陣 (Confusion Matrix):")
print(conf_matrix)

print("模型預測漲的次數:", np.sum(y_pred_class == 1))
print("測試集總筆數:", len(y_test_3d))

final_model.save("2330_model.h5")
scaler_filename = "2330_scaler.pkl"
joblib.dump(scaler_final, scaler_filename)