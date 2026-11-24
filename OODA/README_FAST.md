# Fast-path kit (unzip INTO C:\Users\HP\Desktop\C-V2X-OODA\OODA\ , overwrite when asked)

New / updated files
  ooda_loop.py       chain runner: --observe_mode rules  --act_mode rules  --prompt_lookup N  --compact  --upstream gold  --skip_act  --obs_ids FILE
  observe_rules.py   Observe as deterministic code (verified vs 7,439 dataset records: all categorical fields 98.8-100 %, only rounding noise in trend numbers)
  act_guard.py       hard-case guard (malformed / infeasible policy, spoofed feedback)
  build_act.py       (path fix) its act_teacher() is the deterministic Act used by --act_mode rules
  compact.py         compact one-line formats for Orient and Decide + lossless expanders (round-trip verified on all 7,031 records each)
  build_compact.py   writes data_compact/  (already included)
  eval_compact.py    evaluate a compact adapter: expansion, exact match, tokens and seconds per example
  data_compact\      orient_S*.jsonl, decide_S*.jsonl with compact targets (same splits as data\)

STEP A (10 min) - measure the no-training speed-ups, from ...\Fine-tune:
  .\ft\Scripts\python.exe ..\OODA\ooda_loop.py --data_dir ..\OODA\data --schema_dir ..\OODA\schemas --adapters adapters --n 10 --observe_mode rules --act_mode rules --prompt_lookup 10 --out loop_fast.jsonl
  (if prompt-lookup errors, repeat without --prompt_lookup 10)

STEP B (about 1.5 h each, GPU) - train the compact adapters, warm-started from the JSON adapters, from ...\Fine-tune:
  .\ft\Scripts\python.exe train_lora.py --agent decide --model meta-llama/Llama-3.2-3B-Instruct --data_dir ..\OODA\data_compact --init_adapter adapters\decide_FINAL --out adapters\decide_compact --max_steps 150 --eval_steps 50 --save_steps 50 --eval_max 100 --save_only_model
  .\ft\Scripts\python.exe train_lora.py --agent orient --model meta-llama/Llama-3.2-3B-Instruct --data_dir ..\OODA\data_compact --init_adapter adapters\orient_FINAL --out adapters\orient_compact --max_steps 150 --eval_steps 50 --save_steps 50 --eval_max 100 --save_only_model
  (the best step is picked by validation loss and copied to the top of the output folder; if it is not, copy the last checkpoint-150 contents up)

STEP C (10 min) - evaluate each compact adapter (accuracy after expansion + seconds per example), from ...\Fine-tune:
  .\ft\Scripts\python.exe ..\OODA\eval_compact.py --agent decide --adapter adapters\decide_compact --data_dir ..\OODA\data --compact_dir ..\OODA\data_compact --schema_dir ..\OODA\schemas --n 100 --bsz 1
  .\ft\Scripts\python.exe ..\OODA\eval_compact.py --agent orient --adapter adapters\orient_compact --data_dir ..\OODA\data --compact_dir ..\OODA\data_compact --schema_dir ..\OODA\schemas --n 100 --bsz 1

STEP D (the fast chain) - everything together:
  .\ft\Scripts\python.exe ..\OODA\ooda_loop.py --data_dir ..\OODA\data --schema_dir ..\OODA\schemas --adapters adapters --n 30 --compact --observe_mode rules --act_mode rules --prompt_lookup 10 --out loop_compact.jsonl
