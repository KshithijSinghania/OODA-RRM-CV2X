# Data, results and models (not in this repository)

Datasets, simulation outputs and model weights are too large for Git and are kept out by `.gitignore`.
They are attached to this repository's private GitHub Release **data-v1**.

| What | Folder in the project | Release asset | How to rebuild |
| --- | --- | --- | --- |
| Raw SUMO / ns-3 telemetry (S1-S7) | `datasets/`, `heldout/*.csv` | `datasets.zip` | `python/collect_*.py`, `heldout/make_s5_dataset.py` |
| Agent training data | `OODA*/data*/` | `ooda_data.zip` | `OODA/build_*.py`, `build_compact2.py` |
| Closed-loop ns-3 results | `ns3_exchange/results/` | `ns3_results.zip` | `ns3_exchange/scripts/run_cl.sh` |
| Live decision traces | `Fine-tune/live_logs/` | `live_logs.zip` | `OODA/ooda_loop.py` with ns-3 |
| Enforcer runs | `enforcer/runs*/` | `enforcer_runs.zip` | `enforcer/train_ppo.py`, `train_ctx.py` |
| Final LoRA adapters | `Fine-tune/adapters/orient_cov`, `decide_cov2`, `act_compact2` | `adapters_final.zip` | `Fine-tune/train_lora.py` |
| Merged and GGUF models | `Fine-tune/merged/`, `Fine-tune/gguf/` | not uploaded | `OODA/export_merged.py`, then llama.cpp conversion |

To restore, download the assets and unzip each into the folder named above.
