@echo off
rem ACT round 2: (1) error breakdown of the current Act adapter by record kind, (2) continue training it (it had only seen 0.83 epoch), (3) re-evaluate, (4) chain. Unattended, about 2 hours.
cd /d "%~dp0"
if not exist logs mkdir logs
findstr /c:"adapter_override" "..\OODA\ooda_loop.py" >nul || (echo ERROR: unzip the NEW ooda_infer_kit.zip into ..\OODA first, overwrite all & pause & exit /b 1)
findstr /c:"kind_ok" "..\OODA\eval_compact.py" >nul || (echo ERROR: unzip the NEW ooda_infer_kit.zip into ..\OODA first, overwrite all & pause & exit /b 1)
echo.
echo ===== %date% %time%  1/4 evaluate current ACT with error dump (about 6 min) =====
powershell -NoProfile -Command ".\ft\Scripts\python.exe ..\OODA\eval_compact.py --agent act --adapter adapters\act_compact2 --data_dir ..\OODA\data --compact_dir ..\OODA\data_compact2 --schema_dir ..\OODA\schemas --n 0 --format v2 --bsz 1 --dump logs\act_errors_a.jsonl 2>&1 | Tee-Object -FilePath logs\eval_act_a.log"
echo.
echo ===== %date% %time%  2/4 continue training ACT, 500 more steps (about 95 min) =====
powershell -NoProfile -Command ".\ft\Scripts\python.exe train_lora.py --agent act --model meta-llama/Llama-3.2-3B-Instruct --data_dir ..\OODA\data_compact2 --init_adapter adapters\act_compact2 --out adapters\act_compact2b --max_steps 500 --eval_steps 100 --save_steps 100 --eval_max 100 --save_only_model 2>&1 | Tee-Object -FilePath logs\train_act_b.log"
echo.
echo ===== %date% %time%  3/4 evaluate continued ACT with error dump =====
powershell -NoProfile -Command ".\ft\Scripts\python.exe ..\OODA\eval_compact.py --agent act --adapter adapters\act_compact2b --data_dir ..\OODA\data --compact_dir ..\OODA\data_compact2 --schema_dir ..\OODA\schemas --n 0 --format v2 --bsz 1 --dump logs\act_errors_b.jsonl 2>&1 | Tee-Object -FilePath logs\eval_act_b.log"
echo.
echo ===== %date% %time%  4/4 chain on 30 windows with the continued ACT =====
powershell -NoProfile -Command ".\ft\Scripts\python.exe ..\OODA\ooda_loop.py --data_dir ..\OODA\data --schema_dir ..\OODA\schemas --adapters adapters --adapter_override act=act_compact2b --n 30 --compact2 --observe_mode rules --out loop_compact2_actb.jsonl 2>&1 | Tee-Object -FilePath logs\chain_actb.log"
echo.
echo ===== %date% %time%  ALL DONE =====
echo Send me: logs\eval_act_a.log logs\eval_act_b.log logs\chain_actb.log logs\act_errors_a.jsonl logs\act_errors_b.jsonl logs\train_act_b.log
pause
