@echo off
setlocal
cd /d "%~dp0"
set "VENV_DIR=.venv-sleep"
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo First run: creating an isolated Python environment...
  py -3 -m venv "%VENV_DIR%" || exit /b 1
  "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
  "%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements-sleep-inference.txt || exit /b 1
)
"%VENV_DIR%\Scripts\python.exe" sleep_chat.py %*
if errorlevel 1 pause
