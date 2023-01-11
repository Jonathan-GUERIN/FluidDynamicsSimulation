@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="ordijonat" (taskkill /f /pid 16680)
if /i "%LOCALHOST%"=="ordijonat" (taskkill /f /pid 12076)

del /F cleanup-ansys-ordijonat-12076.bat
