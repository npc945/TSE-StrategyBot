@echo off
chcp 65001 > nul
echo ==========================================
echo 🚀 AI 量化交易系統 - 每日自動化排程啟動
echo ==========================================

:: 切換到程式碼所在的資料夾
cd /d "%~dp0"

echo.
echo [1/3] 正在更新最新股市資料 (stock_finmind.py)...
python stock_finmind.py
if %errorlevel% neq 0 goto error

echo.
echo [2/3] 正在計算技術指標與特徵 (tech.py)...
python tech.py
if %errorlevel% neq 0 goto error

echo.
echo [3/3] 正在執行每日實盤跟進與網頁資料更新 (daily_test.py)...
python daily_test.py
if %errorlevel% neq 0 goto error

echo.
echo ==========================================
echo ✅ 所有排程執行完畢！網頁與系統資料已更新到最新。
echo ==========================================
pause
exit

:error
echo.
echo ❌ 執行過程中發生錯誤！請檢查上方報錯訊息。
pause
exit