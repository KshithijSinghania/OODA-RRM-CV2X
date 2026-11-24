#!/usr/bin/env python3
"""Evaluate a trained adapter on the test split: JSON validity, schema validity, exact match vs reference.
  python eval_adapter.py --agent decide --adapter adapters/decide --data_dir data --schema_dir schemas --n 100
"""
import argparse, glob, json, os
import torch
from jsonschema import validate, ValidationError
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCHEMA = {'observe': 'observe_summary', 'orient': 'orient_context', 'decide': 'decide_policy', 'act': 'act_command'}
ap = argparse.ArgumentParser()
ap.add_argument('--agent', required=True, choices=SCHEMA)
ap.add_argument('--adapter', required=True)
ap.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct')
ap.add_argument('--data_dir', default='data'); ap.add_argument('--schema_dir', default='schemas')
ap.add_argument('--n', type=int, default=100, help='test examples to score (0 = all)')
ap.add_argument('--bsz', type=int, default=4); ap.add_argument('--max_new', type=int, default=700)
ap.add_argument('--dump', default=None, help='write every mismatch (id, kind, differing fields, ref vs pred) to this .jsonl')
a = ap.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(a.data_dir, f'{a.agent}_S?.jsonl'))):
    rows += [json.loads(l) for l in open(f, encoding='utf-8')]
rows = [r for r in rows if r['split'] == 'test']
if a.n: rows = rows[::max(1, len(rows) // a.n)][:a.n]
schema = json.load(open(os.path.join(a.schema_dir, SCHEMA[a.agent] + '.schema.json')))

tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = 'left'
tok.pad_token = tok.pad_token or '<|reserved_special_token_0|>'
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
import transformers
_v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
_dt = {'dtype' if _v >= (4, 56) else 'torch_dtype': torch.bfloat16}
model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map={'': 0}, **_dt)
model = PeftModel.from_pretrained(model, a.adapter).eval()
stop = [tok.eos_token_id, tok.convert_tokens_to_ids('<|eot_id|>')]

n = ok_json = ok_schema = exact = 0
bykind = {}; dumpf = open(a.dump, 'w', encoding='utf-8') if a.dump else None
key_hit, key_tot = {}, {}
for i in range(0, len(rows), a.bsz):
    b = rows[i:i + a.bsz]
    prompts = [tok.apply_chat_template(r['messages'][:2], tokenize=False, add_generation_prompt=True) for r in b]
    enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, eos_token_id=stop, pad_token_id=tok.pad_token_id)
    for r, o in zip(b, out):
        txt = tok.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        ref = json.loads(r['messages'][2]['content']); n += 1
        kd = str(r.get('kind', '-')).split(':')[0]; bykind.setdefault(kd, [0, 0]); bykind[kd][1] += 1
        try: pred = json.loads(txt)
        except Exception:
            if dumpf: dumpf.write(json.dumps({'id': r['id'], 'kind': r.get('kind'), 'error': 'invalid_json', 'raw': txt[:1500]}) + '\n')
            continue
        ok_json += 1
        serr = None
        try: validate(pred, schema); ok_schema += 1
        except ValidationError as e: serr = e.message[:300]
        exact += pred == ref; bykind[kd][0] += (pred == ref)
        if dumpf and (pred != ref or serr):
            diff = {k: {'ref': ref.get(k), 'pred': pred.get(k)} for k in ref if isinstance(pred, dict) and pred.get(k) != ref.get(k)} if isinstance(ref, dict) else {}
            dumpf.write(json.dumps({'id': r['id'], 'kind': r.get('kind'), 'detail': r.get('detail'), 'schema_error': serr, 'diff': diff}) + '\n')
        if isinstance(pred, dict) and isinstance(ref, dict):
            for k, v in ref.items():
                key_tot[k] = key_tot.get(k, 0) + 1; key_hit[k] = key_hit.get(k, 0) + (pred.get(k) == v)
    print(f'  scored {n}/{len(rows)}', end='\r', flush=True)
print(f'\n{a.agent}: n={n}  valid JSON {ok_json/n:.1%}  schema-valid {ok_schema/n:.1%}  exact match {exact/n:.1%}')
for k in sorted(key_tot, key=lambda k: key_hit[k] / key_tot[k]):
    print(f'  field {k:<28} {key_hit[k]/key_tot[k]:.1%}')

if len(bykind) > 1 or '-' not in bykind:
    print('  exact match by kind:')
    for k, (h, t) in sorted(bykind.items()): print(f'    {k:<24} {h}/{t}  ({h/t:.0%})')
if dumpf: dumpf.close(); print('mismatches written to', a.dump)
