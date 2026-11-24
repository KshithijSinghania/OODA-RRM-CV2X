#!/usr/bin/env python3
"""Merge one LoRA adapter into the full-precision base model and save a plain Hugging Face folder (input for llama.cpp's convert_hf_to_gguf.py).
Runs on CPU (needs about 8 GB of free RAM, 3-5 minutes). Run from ...\\Fine-tune:
  .\\ft\\Scripts\\python.exe ..\\OODA\\export_merged.py --adapter adapters\\decide_compact2 --out merged\\decide_compact2
NOTE: the adapters were trained on a 4-bit (NF4) base; merging into the 16-bit base and quantising again is standard practice but changes the weights slightly.
Always re-run eval_compact.py through the llama-server and compare with the Hugging Face numbers (README_INFER.md, step 6)."""
import argparse, os, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument('--adapter', required=True); ap.add_argument('--out', required=True); ap.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct')
a = ap.parse_args()
base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={'': 'cpu'})
m = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
os.makedirs(a.out, exist_ok=True); m.save_pretrained(a.out, safe_serialization=True); AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
print('merged model saved to', a.out)
