@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
rem Prefer "python": the launchers and README run "python live_caption_gemini.py"
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if not errorlevel 1 goto use_python
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if not errorlevel 1 goto use_py
echo Python 3.10 or newer was not found.
echo Download 64-bit Python from python.org.
echo During installation, check "Add python.exe to PATH".
echo Then double-click install.bat again.
start "" "https://www.python.org/downloads/windows/"
pause
exit /b 1

:use_py
py -3 install.py
set "INSTALL_RESULT=%ERRORLEVEL%"
goto done

:use_python
python install.py
set "INSTALL_RESULT=%ERRORLEVEL%"

:done
pause
exit /b %INSTALL_RESULT%
