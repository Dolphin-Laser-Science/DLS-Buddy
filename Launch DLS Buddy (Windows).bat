@echo off
cd /d "%~dp0"

rem First run: create the virtual environment. A failed create must not leave a
rem half-made .venv behind, or every later launch would skip this step and fail.
if not exist .venv\Scripts\python.exe (
    echo Setting up environment for the first time, this may take a minute...
    py -3.13 -m venv .venv
    if errorlevel 1 (
        echo.
        echo Could not create the virtual environment. Is Python 3.13 installed
        echo and on PATH? https://www.python.org/downloads/
        if exist .venv rmdir /s /q .venv
        pause
        exit /b 1
    )
)

rem Install (or re-install) dependencies whenever requirements.txt has changed
rem since the last successful install, so an update that adds a package works.
set "STAMP=.venv\requirements.installed"
set NEED_INSTALL=0
if not exist "%STAMP%" set NEED_INSTALL=1
if exist "%STAMP%" fc /b requirements.txt "%STAMP%" >nul 2>&1 || set NEED_INSTALL=1
if %NEED_INSTALL%==1 (
    echo Installing dependencies...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency installation failed. Check your network connection and
        echo run this launcher again.
        pause
        exit /b 1
    )
    copy /y requirements.txt "%STAMP%" >nul
)

.venv\Scripts\python.exe -m gui.main
if errorlevel 1 (
    echo.
    echo DLS Buddy exited with an error ^(see the messages above^).
    pause
)
