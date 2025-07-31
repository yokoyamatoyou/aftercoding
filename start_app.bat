@echo off
REM アンケート分析ツール GUI 起動スクリプト
REM このスクリプトをダブルクリックするとGUIが起動します。

cd /d %~dp0coding\survey_analysis_mvp
python main.py
pause
