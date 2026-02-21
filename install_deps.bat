@echo off
echo ==========================================
echo   INSTALLING DEPENDENCIES
echo ==========================================
echo.

pip install sounddevice numpy pedalboard keyboard scipy customtkinter pystray Pillow psutil pynput python-dotenv requests

echo.
echo Done! You can now run:
echo   run_basic.bat    - Simple push-to-talk voice changer
echo   run_advanced.bat - Full GUI with recording and realtime testing
echo.
pause
