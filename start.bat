@echo off
REM ==== FOOTBALL ORACLE - khoi dong nhanh ====
REM Bam dup file nay de tu cai thu vien va chay tool.
cd /d "%~dp0"

echo [1/2] Cai thu vien can thiet...
python -m pip install --quiet fastapi uvicorn httpx python-dotenv
if errorlevel 1 (
  echo.
  echo LOI: chua co Python. Hay cai Python tai https://python.org va tich "Add Python to PATH".
  pause
  exit /b 1
)

echo [2/2] Khoi dong server tai http://127.0.0.1:8000
echo (Mo trinh duyet vao dia chi tren. Dong cua so nay de tat tool.)
echo.
REM Mo trinh duyet tu dong sau 2 giay
start "" cmd /c "timeout /t 2 >nul & start http://127.0.0.1:8000"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
pause
