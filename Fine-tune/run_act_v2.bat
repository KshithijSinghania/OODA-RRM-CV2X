@echo off
rem Evaluates the code-based OBSERVE, trains ACT (token-diet v2), evaluates it, then runs the chain with Observe = rules (code), Orient/Decide/Act = LLMs. Unattended. Put in C:\Users\HP\Desktop\C-V2X-OODA\Fine-tune and double-click.
cd /d "%~dp0"
if not exist logs mkdir logs
if not exist "..\OODA\data_compact2\act_S2.jsonl" (echo ERROR: ..\OODA\data_compact2\act_S2.jsonl missing - unzip ooda_infer_kit.zip into ..\OODA first & pause & exit /b 1)
if not exist "..\OODA\eval_observe_rules.py" (echo ERROR: ..\OODA\eval_observe_rules.py missing - unzip ooda_infer_kit.zip into ..\OODA first & pause & exit /b 1)
findstr /c:"decode_act2" "..\OODA\eval_compact.py" >nul || (echo ERROR: ..\OODA\eval_compact.py is the OLD version - overwrite it from ooda_infer_kit.zip & pause & exit /b 1)
findstr /c:"act_input2" "..\OODA\ooda_loop.py" >nul || (echo ERROR: ..\OODA\ooda_loop.py is the OLD version - overwrite it from ooda_infer_kit.zip & pause & exit /b 1)
echo.
echo ===== %date% %time%  0/3 evaluate OBSERVE (rules/code, no GPU, seconds) =====
powershell -NoProfile -Command ".\ft\Scripts\python.exe ..\OODA\eval_observe_rules.py --data_dir ..\OODA\data --schema_dir ..\OODA\schemas 2>&1 | Tee-Object -FilePath logs\eval_observe_rules.log"
echo.
echo ===== %date% %time%  1/3 train ACT (400 steps) =====
powershell -NoProfile -Command ".\ft\Scripts\python.exe train_lora.py --agent act --model meta-llama/Llama-3.2-3B-Instruct --data_dir ..\OODA\data_compact2 --init_adapter adapters\act_FINAL --out adapters\act_compact2 --max_steps 400 --eval_steps 100 --save_steps 100 --eval_max 100 --save_only_model 2>&1 | Tee-Object -FilePath logs\train_act.log"
echo.
echo ===== %date% %time%  2/3 evaluate ACT =====
powershell -NoProfile -Command ".\ft\Scripts\python.exe ..\OODA\eval_compact.py --agent act --adapter adapters\act_compact2 --data_dir ..\OODA\data --compact_dir ..\OODA\data_compact2 --schema_dir ..\OODA\schemas --n 0 --format v2 --bsz 1 2>&1 | Tee-Object -FilePath logs\eval_act.log"
echo.
echo ===== %date% %time%  3/3 chain on 30 windows (Observe=code, Orient/Decide/Act=LLM) =====
powershell -NoProfile -Command ".\ft\Scripts\python.exe ..\OODA\ooda_loop.py --data_dir ..\OODA\data --schema_dir ..\OODA\schemas --adapters adapters --n 30 --compact2 --observe_mode rules --out loop_compact2_act.jsonl 2>&1 | Tee-Object -FilePath logs\chain_act.log"
echo.
echo ===== %date% %time%  ALL DONE =====
echo Send me: logs\eval_observe_rules.log, logs\eval_act.log, logs\chain_act.log, loop_compact2_act.jsonl (and logs\train_act.log)
pause
