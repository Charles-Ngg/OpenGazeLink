@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Run setup-venv.bat first.
  exit /b 1
)
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
".venv\Scripts\python.exe" -m opengazelink_pc runtime
