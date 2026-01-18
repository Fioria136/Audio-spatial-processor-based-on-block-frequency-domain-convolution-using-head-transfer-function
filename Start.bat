@echo off
REM ===============================
REM BPM 分析批处理启动脚本
REM ===============================

REM 切换到脚本所在目录
cd /d %~dp0

REM 运行 Python 脚本
Run.py

REM 等待用户按任意键退出
pause