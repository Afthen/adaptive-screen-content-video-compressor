@echo off
if "%~1"=="" (
    echo Please drag and drop a video file directly onto this script.
    pause
    exit /b
)

:: Passes the raw dragged filepath accurately into your dedicated execution Powershell context natively
PowerShell -NoProfile -ExecutionPolicy Bypass -File "%~dp0archiver_pipeline.ps1" -InputFile "%~1"