@echo off
cd /d "%~dp0"
if not exist .venv ( py -3 -m venv .venv )
.venv\Scripts\python.exe -m pip install -q -r requirements.txt
echo Opening Blue Voice at http://127.0.0.1:8765  (close this window to stop it)
.venv\Scripts\python.exe main.py
pause
