@echo off
setlocal
cd /d "%~dp0"
title Camera Web App

set "VENV=.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\requirements.installed"

if exist "%VPY%" goto have_venv

set "PYCMD="
where py >nul 2>nul
if not errorlevel 1 set "PYCMD=py -3"
if not defined PYCMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PYCMD=python"
)
if not defined PYCMD (
    echo Python 3 was not found. Install it from https://www.python.org and retry.
    pause
    exit /b 1
)

echo Creating virtual environment...
%PYCMD% -m venv "%VENV%"
if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)
goto check_deps

:have_venv
echo Using existing virtual environment.

:check_deps
set "NEEDINSTALL=0"
if not exist "%STAMP%" set "NEEDINSTALL=1"
if exist "%STAMP%" (
    fc /b requirements.txt "%STAMP%" >nul 2>nul
    if errorlevel 1 set "NEEDINSTALL=1"
)

if "%NEEDINSTALL%"=="1" (
    echo Installing dependencies...
    "%VPY%" -m pip install --upgrade pip >nul 2>nul
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed.
        pause
        exit /b 1
    )
    copy /y requirements.txt "%STAMP%" >nul
)

netsh advfirewall firewall show rule name="Camera Web App 8443" >nul 2>nul
if errorlevel 1 (
    netsh advfirewall firewall add rule name="Camera Web App 8443" dir=in action=allow protocol=TCP localport=8443 profile=private,public >nul 2>nul
    if errorlevel 1 (
        echo.
        echo Could not open firewall port 8443, admin rights are needed.
        echo Phones will not connect until you run this once in an admin prompt:
        echo netsh advfirewall firewall add rule name="Camera Web App 8443" dir=in action=allow protocol=TCP localport=8443 profile=private,public
        echo.
    ) else (
        echo Firewall rule added for port 8443.
    )
)

"%VPY%" app.py

echo.
echo Server stopped.
pause
