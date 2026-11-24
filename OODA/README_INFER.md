# Inference kit: from 4.4 s to about 1 s with no new software, and below that with llama.cpp

Unzip INTO C:\Users\HP\Desktop\C-V2X-OODA\OODA\ (overwrite when asked). Run commands from ...\Fine-tune unless stated.

## Where the time goes (measured on YOUR traces: `fit_latency.py`)
seconds per Orient/Decide hop = 0.19 (fixed) + 0.00045 x prompt tokens + 0.073 x generated tokens  (R2 0.89, 110 windows)
-> decode 13.7 tokens/s, prefill 2,200 tokens/s. A typical hop (850 prompt, 27 generated tokens) spends 76 % decoding, 15 % prefill, 9 % fixed.

## What is new: token diet v2 (compact2.py)
The labels are exact functions of a handful of input fields, and the outputs carry only about 12 bits each (checked on all 7,031 records: zero conflicts).
So v2 feeds the model ONLY the fields the teacher code reads (65-82 characters instead of 1,300-1,900) and asks for a 6-digit (Orient) or 12-digit (Decide) code.
Digits are tokenised in chunks of three, so Orient writes 2 tokens and Decide 4 (plus end-of-text). Expansion back to the full JSON is exact (verified on every record).
ESTIMATE with the fitted model: about 1.1 s per window on the Hugging Face path you already have (was 4.4 s). NOT measured until you run step 3.

## Step 1 (GPU, about 30-60 min each; sequences are ~10x shorter than before): train the v2 adapters
.\ft\Scripts\python.exe train_lora.py --agent orient --model meta-llama/Llama-3.2-3B-Instruct --data_dir ..\OODA\data_compact2 --init_adapter adapters\orient_FINAL --out adapters\orient_compact2 --max_steps 300 --eval_steps 100 --save_steps 100 --eval_max 100 --save_only_model
.\ft\Scripts\python.exe train_lora.py --agent decide --model meta-llama/Llama-3.2-3B-Instruct --data_dir ..\OODA\data_compact2 --init_adapter adapters\decide_FINAL --out adapters\decide_compact2 --max_steps 300 --eval_steps 100 --save_steps 100 --eval_max 100 --save_only_model
(If an adapter folder has no adapter_model.safetensors at its top level, copy the contents of the last checkpoint-* folder up one level.)

## Step 2 (about 5 min each): accuracy, with the seen / unseen split
.\ft\Scripts\python.exe ..\OODA\eval_compact.py --agent orient --adapter adapters\orient_compact2 --data_dir ..\OODA\data --compact_dir ..\OODA\data_compact2 --schema_dir ..\OODA\schemas --n 0 --format v2 --bsz 1
.\ft\Scripts\python.exe ..\OODA\eval_compact.py --agent decide --adapter adapters\decide_compact2 --data_dir ..\OODA\data --compact_dir ..\OODA\data_compact2 --schema_dir ..\OODA\schemas --n 0 --format v2 --bsz 1
READ THE "exact match by input novelty" LINE. 85 % of Orient and 96 % of Decide test prompts also occur in training, so a high overall score is largely lookup;
the "unseen" figure (76 Orient and 21 Decide test records) is the only evidence of generalisation, and it is small.

## Step 3 (about 2 min): the chain, measured
.\ft\Scripts\python.exe ..\OODA\ooda_loop.py --data_dir ..\OODA\data --schema_dir ..\OODA\schemas --adapters adapters --n 30 --compact2 --observe_mode rules --act_mode rules --out loop_compact2.jsonl
(no --prompt_lookup: with 2-4 generated tokens it cannot help). Compare "total ... s per cell-window" with 4.4 s.
Then:  .\ft\Scripts\python.exe ..\OODA\fit_latency.py loop_compact2.jsonl

## Step 4 (optional, for below 1 s): llama.cpp
Everything here is written against the documented llama-server API and tested only against a stand-in server (test_llama_engine.py); it has NOT been run against a real llama.cpp. Flag names change between llama.cpp versions: check `llama-server.exe --help`.
a) Download a llama.cpp release for Windows with CUDA (llama-b*-bin-win-cuda-*-x64.zip and the matching cudart zip) from https://github.com/ggml-org/llama.cpp/releases and unzip both to C:\llama.cpp\ ;
   download the same release's "Source code (zip)" and keep convert_hf_to_gguf.py (and its gguf-py folder); then:  .\ft\Scripts\python.exe -m pip install gguf sentencepiece
b) merge each adapter into the base model (CPU, about 5 min, needs about 8 GB free RAM):
   .\ft\Scripts\python.exe ..\OODA\export_merged.py --adapter adapters\orient_compact2 --out merged\orient_compact2
   .\ft\Scripts\python.exe ..\OODA\export_merged.py --adapter adapters\decide_compact2 --out merged\decide_compact2
c) convert and quantise (repeat for decide; use Q8_0 instead of Q4_K_M if accuracy drops in step e):
   .\ft\Scripts\python.exe C:\llama.cpp\src\convert_hf_to_gguf.py merged\orient_compact2 --outfile gguf\orient_compact2-f16.gguf --outtype f16
   C:\llama.cpp\llama-quantize.exe gguf\orient_compact2-f16.gguf gguf\orient_compact2-Q4_K_M.gguf Q4_K_M
d) speed of your GPU:  C:\llama.cpp\llama-bench.exe -m gguf\orient_compact2-Q4_K_M.gguf -ngl 99 -p 512 -n 128
   then:  .\ft\Scripts\python.exe ..\OODA\fit_latency.py loop_compact2.jsonl --v2 --cfg llama3b_q4:<pp512 t/s>:<tg128 t/s>
e) servers (two terminals; ports 8081 / 8082):
   C:\llama.cpp\llama-server.exe -m gguf\orient_compact2-Q4_K_M.gguf -ngl 99 -c 1024 --port 8081
   C:\llama.cpp\llama-server.exe -m gguf\decide_compact2-Q4_K_M.gguf -ngl 99 -c 1024 --port 8082
   accuracy parity with the Hugging Face numbers of step 2 (the first request prints a tokenisation check; it must say "identical"):
   .\ft\Scripts\python.exe ..\OODA\eval_compact.py --agent decide --engine llama --url http://127.0.0.1:8082 --data_dir ..\OODA\data --compact_dir ..\OODA\data_compact2 --schema_dir ..\OODA\schemas --n 0 --format v2
f) the chain through the servers:
   .\ft\Scripts\python.exe ..\OODA\ooda_loop.py --data_dir ..\OODA\data --schema_dir ..\OODA\schemas --n 30 --compact2 --observe_mode rules --act_mode rules --engine llama --llama_urls orient=http://127.0.0.1:8081,decide=http://127.0.0.1:8082 --out loop_llama.jsonl

## What this does NOT do
- Merged-and-quantised models can differ from the QLoRA-trained ones: step 4e is the check.
- The model no longer sees the retrieved RAG snippets, the narrative or unused telemetry (rag_refs, rationale, confidence are produced by code). Keep the full-prompt adapters as the reference model.
- Reaching 200 ms for Orient + Decide together needs both llama.cpp and a fast enough GPU; on the A1000 it is plausible but unmeasured.

## Step 1b: Observe and Act as fine-tuned LLMs too (all four agents, same token diet)
What the model does now (everything else is filled in by code from the full input, exactly as for Orient/Decide):
- Observe: reads ~100 characters (the fields the summary rules read; the two trend deltas are computed by code) and writes 9 digits = load level, QoS severity, QoS trend, load trend, 5 anomaly flags (3 tokens).
- Act: reads ~290 characters (policy, constraints, enforcer feedback) and writes 2 digits = escalation reason, policy-rejected flag (1 token). The A1/E2 translation is a template from the policy.
Checked offline on every record: zero round-trip mismatches, and zero conflicts (same prompt, different label). UNLIKE Orient/Decide these prompts are almost never seen in training
(Observe 4.8 % of test prompts, Act 0 % because policy ids are random), so the test score below is genuine generalisation, not lookup.
ESTIMATE: +0.25-0.3 s per extra hop on the HF path -> about 1.2-1.4 s per window with four LLM hops (NOT measured for four hops; the measured configuration is Observe = code + three LLM hops = 0.9 s, see step 3c).

Train (about 30-60 min each; the sequences are short):
.\ft\Scripts\python.exe train_lora.py --agent observe --model meta-llama/Llama-3.2-3B-Instruct --data_dir ..\OODA\data_compact2 --init_adapter adapters\observe_FINAL --out adapters\observe_compact2 --max_steps 600 --eval_steps 100 --save_steps 100 --eval_max 100 --save_only_model
.\ft\Scripts\python.exe train_lora.py --agent act --model meta-llama/Llama-3.2-3B-Instruct --data_dir ..\OODA\data_compact2 --init_adapter adapters\act_FINAL --out adapters\act_compact2 --max_steps 400 --eval_steps 100 --save_steps 100 --eval_max 100 --save_only_model

Step 2b: accuracy (5 min each). Observe is graded on the whole summary (categorical fields from the model; copies and narrative from code), so numeric trend deltas can differ in the last digit - read the per-field lines:
.\ft\Scripts\python.exe ..\OODA\eval_compact.py --agent observe --adapter adapters\observe_compact2 --data_dir ..\OODA\data --compact_dir ..\OODA\data_compact2 --schema_dir ..\OODA\schemas --n 0 --format v2 --bsz 1
.\ft\Scripts\python.exe ..\OODA\eval_compact.py --agent act --adapter adapters\act_compact2 --data_dir ..\OODA\data --compact_dir ..\OODA\data_compact2 --schema_dir ..\OODA\schemas --n 0 --format v2 --bsz 1

Step 3b (optional, NOT run so far): the chain with all four agents as LLMs (needs observe_compact2 trained, Step 1b; no --observe_mode rules / --act_mode rules):
.\ft\Scripts\python.exe ..\OODA\ooda_loop.py --data_dir ..\OODA\data --schema_dir ..\OODA\schemas --adapters adapters --n 30 --compact2 --out loop_compact2_all4.jsonl

Step 3c (the configuration actually measured, 0.9 s per window): Observe = code (observe_rules.py, 97.9 % on its 5 categorical decisions), Orient + Decide + Act = fine-tuned LLMs:
.\ft\Scripts\python.exe ..\OODA\ooda_loop.py --data_dir ..\OODA\data --schema_dir ..\OODA\schemas --adapters adapters --n 30 --compact2 --observe_mode rules --out loop_compact2_act.jsonl
