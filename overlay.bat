@echo off
cd /d "%~dp0"
if not exist .venv ( py -3 -m venv .venv )
.venv\Scripts\python.exe -m pip install -q -r requirements-overlay.txt
echo Overlay mode: only the three buttons and the message box float over your screen.
.venv\Scripts\python.exe overlay.py
pause
