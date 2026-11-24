#!/bin/bash
# Trains the four agent adapters one after another on Llama-3.2-3B-Instruct. Safe to re-run:
# finished agents are skipped, interrupted ones resume from their last checkpoint.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=${MODEL:-meta-llama/Llama-3.2-3B-Instruct}
for a in observe act orient decide; do
  out=adapters/$a
  if [ -f $out/adapter_model.safetensors ]; then echo "[$a] already trained, skipping"; continue; fi
  extra=""; ls $out/checkpoint-* >/dev/null 2>&1 && extra="--resume"
  echo "[$a] $(date) starting $extra"
  python train_lora.py --agent $a --data_dir data --model $MODEL --out $out \
      --bsz 2 --ga 8 --epochs 3 --eval_steps 200 $extra 2>&1 | tee -a logs_$a.txt || { echo "[$a] FAILED"; exit 1; }
done
echo "ALL DONE $(date)"
