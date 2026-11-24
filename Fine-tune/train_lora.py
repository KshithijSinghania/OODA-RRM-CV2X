#!/usr/bin/env python3
"""QLoRA fine-tuning of one OODA agent adapter (observe | orient | decide | act).
Proposal settings: LoRA rank 32, alpha 64, Llama-3-8B, one adapter per agent.
Loss is computed on the assistant reply only (prompt tokens are masked).

  python train_lora.py --agent observe --data_dir data --out adapters/observe
Resume after an interruption: add --resume (continues from the latest checkpoint in --out).
"""
import argparse, glob, json, os, random, sys
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq, set_seed)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ap = argparse.ArgumentParser()
ap.add_argument('--agent', required=True, choices=['observe', 'orient', 'decide', 'act'])
ap.add_argument('--data_dir', default='data')
ap.add_argument('--model', default='meta-llama/Meta-Llama-3-8B-Instruct')
ap.add_argument('--out', default=None)
ap.add_argument('--epochs', type=float, default=3)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--max_len', type=int, default=1536)       # longest example is ~1,300 tokens
ap.add_argument('--bsz', type=int, default=1)               # per-device batch; raise on a big GPU
ap.add_argument('--ga', type=int, default=16)               # gradient accumulation -> effective batch = bsz*ga
ap.add_argument('--rank', type=int, default=32)
ap.add_argument('--alpha', type=int, default=64)
ap.add_argument('--dropout', type=float, default=0.05)
ap.add_argument('--eval_steps', type=int, default=100)
ap.add_argument('--max_train', type=int, default=0, help='debug: use only N training examples')
ap.add_argument('--max_val', type=int, default=0, help='debug: use only N validation examples')
ap.add_argument('--resume', action='store_true')
ap.add_argument('--optim', default='adamw_bnb_8bit', help="optimizer; 'paged_adamw_8bit' was used before (suspected cause of silent crashes at checkpoint save)")
ap.add_argument('--save_only_model', action='store_true', help='checkpoints keep adapter weights only (no optimizer state): cannot --resume, continue with --init_adapter')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--save_steps', type=int, default=0, help='checkpoint every N steps (default: = --eval_steps)')
ap.add_argument('--eval_max', type=int, default=0, help='use only N val examples for the in-training evals (faster)')
ap.add_argument('--init_adapter', default=None, help='warm start: load LoRA weights from this folder (fresh optimizer)')
ap.add_argument('--max_steps', type=int, default=0, help='train exactly N optimizer steps (overrides --epochs)')
a = ap.parse_args()
a.out = a.out or f'adapters/{a.agent}'
set_seed(a.seed)

tok = AutoTokenizer.from_pretrained(a.model)
tok.pad_token = tok.pad_token or '<|reserved_special_token_0|>'
EOT = '<|eot_id|>'

def load(split):
    rows = []
    for f in sorted(glob.glob(os.path.join(a.data_dir, f'{a.agent}_S?.jsonl'))):
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            if r['split'] == split: rows.append(r['messages'])
    return rows

class SFT(Dataset):
    def __init__(self, rows):
        self.items, self.cut = [], 0
        for m in rows:
            assert [x['role'] for x in m] == ['system', 'user', 'assistant'], 'unexpected message roles'
            prompt = tok.apply_chat_template(m[:2], tokenize=False, add_generation_prompt=True)
            p = tok(prompt, add_special_tokens=False)['input_ids']          # template already has BOS
            c = tok(m[2]['content'] + EOT, add_special_tokens=False)['input_ids']
            ids = (p + c)[:a.max_len]
            if len(p) + len(c) > a.max_len: self.cut += 1
            labels = ([-100] * len(p) + c)[:a.max_len]
            self.items.append({'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

train_rows, val_rows = load('train'), load('val')
if a.max_train: random.Random(a.seed).shuffle(train_rows); train_rows = train_rows[:a.max_train]
if a.max_val: val_rows = val_rows[:a.max_val]
train, val = SFT(train_rows), SFT(val_rows)
print(f'{a.agent}: train {len(train)}  val {len(val)}  truncated {train.cut}/{val.cut}', flush=True)
if train.cut: print('WARNING: some examples exceed --max_len and were truncated; raise --max_len', file=sys.stderr)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
import transformers
_v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
_dt = {'dtype' if _v >= (4, 56) else 'torch_dtype': torch.bfloat16}
model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map={'': 0},
                                             attn_implementation='sdpa', **_dt)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
if a.init_adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    print('warm start from', a.init_adapter, flush=True)
else:
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias='none', task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
model.print_trainable_parameters()

import math
STEPS = a.max_steps or math.ceil(len(train) / (a.bsz * a.ga) * a.epochs)
WARMUP = 10 if a.init_adapter else max(1, int(0.03 * STEPS))      # explicit steps: works on old and new transformers
print(f'optimizer steps: {STEPS}  warmup: {WARMUP}', flush=True)
args = TrainingArguments(
    output_dir=a.out, num_train_epochs=a.epochs, max_steps=(a.max_steps or -1), learning_rate=a.lr, lr_scheduler_type='cosine', warmup_steps=WARMUP,
    per_device_train_batch_size=a.bsz, per_device_eval_batch_size=a.bsz, gradient_accumulation_steps=a.ga,
    gradient_checkpointing=True, bf16=True, optim=a.optim, save_only_model=a.save_only_model, weight_decay=0.0, max_grad_norm=1.0,
    logging_steps=10, eval_strategy='steps', eval_steps=a.eval_steps, save_strategy='steps', save_steps=(a.save_steps or a.eval_steps),
    save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
    report_to='none', seed=a.seed, dataloader_num_workers=0)

import gc
from transformers import TrainerCallback
class FreeGPU(TrainerCallback):
    """Release cached GPU memory right after each eval and before each checkpoint save (stalls happened exactly there on the 8 GB card)."""
    def _free(self):
        gc.collect(); torch.cuda.empty_cache()
        f, t = torch.cuda.mem_get_info(); print(f'  [gpu] free {f/2**30:.2f} / {t/2**30:.2f} GiB', flush=True)
    def on_evaluate(self, args, state, control, **kw): self._free()
    def on_save(self, args, state, control, **kw): self._free()
tr = Trainer(model=model, args=args, train_dataset=train, eval_dataset=(torch.utils.data.Subset(val, range(min(a.eval_max, len(val)))) if a.eval_max else val),
             data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8), callbacks=[FreeGPU()])
if a.resume:
    for d in sorted(glob.glob(os.path.join(a.out, 'checkpoint-*'))):
        if not (os.path.exists(os.path.join(d, 'trainer_state.json')) and os.path.getsize(os.path.join(d, 'optimizer.pt')) > 0):
            print('ignoring incomplete checkpoint', d, flush=True)
            os.rename(d, d + '.incomplete')
import re
_has_ckpt = any(re.fullmatch(r'checkpoint-\d+', os.path.basename(d)) for d in glob.glob(os.path.join(a.out, 'checkpoint-*'))) if a.resume else False
if a.resume and not _has_ckpt: print('no complete checkpoint found: starting from scratch', flush=True)
tr.train(resume_from_checkpoint=True if _has_ckpt else None)
tr.save_model(a.out); tok.save_pretrained(a.out)
print('saved adapter to', a.out)
