@echo off
echo ========================================
echo   ShipCast - AI Shipping Predictor
echo ========================================
echo.
echo Starting backend server...
echo.

cd /d "%~dp0backend"

REM Install dependencies if needed
echo Checking dependencies...
pip install -q -r requirements.txt 2>nul

echo.
echo Starting FastAPI server on http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo.

python app.py
