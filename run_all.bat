@echo off
chcp 65001 > nul
echo ==========================================
echo 🚀 量化交易系統 - 每日自動化排程啟動
echo ==========================================

:: 切換到程式碼所在的資料夾
cd /d "%~dp0"

echo.
echo [1/5] 正在更新最新股市資料 (stock_finmind.py)...
python stock_finmind.py
if %errorlevel% neq 0 goto error

echo.
echo [2/5] 正在計算技術指標與特徵 (tech.py)...
python tech.py
if %errorlevel% neq 0 goto error

echo.
echo [3/5] 正在執行 AI 核心組每日實盤跟進 (daily_test.py)...
python daily_test.py
if %errorlevel% neq 0 goto error

echo.
echo [4/5] 正在執行傳統對照組每日實盤跟進 (control_backtest.py)...
python control_backtest.py
if %errorlevel% neq 0 goto error

:: ==========================================
:: 第五階段：自動推播至 GitHub 更新網頁
:: ==========================================
echo.
echo [5/5] 準備將最新資料推送至 GitHub 雲端...
git add .
git commit -m "Auto-update daily live data %date%"
git push
if %errorlevel% neq 0 goto error

echo.
echo ==========================================
echo ✅ 所有任務與雲端網頁更新皆已順利執行完畢！
echo ==========================================
pause
exit

:error
echo.
echo ❌ 執行過程中發生錯誤！請檢查上方報錯訊息。
pause
exit