@echo off
REM ============================================================
REM  DoodleBook launcher — SINGLE instance, FIXED port.
REM  Double-click this file to start.  Close this window to quit.
REM
REM  Why this matters: Gradio is given an explicit port. If that
REM  port is still held (a killed instance's socket in TIME_WAIT,
REM  an orphaned python, or app.py/test_final.py running too),
REM  Gradio CRASHES on startup instead of picking another port.
REM  The old loop then relaunched into the same crash forever and
REM  the browser showed "Connection lost. Attempting reconnection".
REM  So: free the port FIRST, then launch.
REM ============================================================
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set DOODLEBOOK_PORT=7880

:loop
echo.
echo === Freeing port %DOODLEBOOK_PORT% (killing any old/stray instance) ===
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%DOODLEBOOK_PORT% " ^| findstr LISTENING') do (
    echo   killing PID %%a holding port %DOODLEBOOK_PORT%
    taskkill /f /pid %%a >nul 2>&1
)

echo === Starting DoodleBook on http://127.0.0.1:%DOODLEBOOK_PORT%/ ===
echo === Open EXACTLY that URL in your browser (not 7860/7870) ===
python run_modal.py
echo.
echo === Server stopped (exit code %errorlevel%). Restarting in 3 seconds... (close this window to quit) ===
timeout /t 3 /nobreak >nul
goto loop
