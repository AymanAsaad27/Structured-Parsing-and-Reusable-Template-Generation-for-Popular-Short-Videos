@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please set up the project environment first. See docs\studio_demo.md.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8765 in your browser.
".venv\Scripts\python.exe" -m beauty_video_templates.demo.app
pause
