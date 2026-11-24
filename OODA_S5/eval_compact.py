#!/usr/bin/env python3
"""Evaluate a COMPACT-format adapter (orient or decide) on the test split.  The generated one-line text is expanded to the full JSON by compact.py and compared with the
original gold JSON; also reports raw-string exact match, tokens generated and seconds per example.
  python eval_compact.py --agent decide --adapter adapters/decide_compact --data_dir data --compact_dir data_compact --schema_dir schemas --n 100 --bsz 1"""
import argparse, collections, glob, json, os, sys, time
import torch
from jsonschema import validate, ValidationError
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compact as K
import compact2 as C2
SCHEMA = {'observe': 'observe_summary', 'orient': 'orient_context', 'decide': 'decide_policy', 'act': 'act_command'}
ap = argparse.ArgumentParser(); ap.add_argument('--agent', required=True, choices=SCHEMA); ap.add_argument('--adapter', default=None, help='adapter folder (not needed with --engine llama)')
ap.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct'); ap.add_argument('--data_dir', default='data'); ap.add_argument('--compact_dir', default='data_compact')
ap.add_argument('--schema_dir', default='schemas'); ap.add_argument('--n', type=int, default=100); ap.add_argument('--bsz', type=int, default=1); ap.add_argument('--max_new', type=int, default=96)
ap.add_argument('--format', default='v1', choices=['v1', 'v2'], help="v2 = token-diet codes (compact2.py); use with --compact_dir ..\\OODA\\data_compact2")
ap.add_argument('--engine', default='hf', choices=['hf', 'llama']); ap.add_argument('--url', default='http://127.0.0.1:8081', help='llama-server URL for --engine llama (this agent\'s server)')
ap.add_argument('--novelty_dir', default=None, help='folder with the data_compact2 files (flags which test prompts also occur in training); default: data_compact2 next to --compact_dir')
ap.add_argument('--guard', action='store_true', help='act only: apply the deterministic Act guard (act_guard.py) to the model output before scoring, as ooda_loop.py does by default')
ap.add_argument('--dump', default=None); ap.add_argument('--prompt_lookup', type=int, default=0)
a = ap.parse_args()
if a.engine == 'hf' and not a.adapter: sys.exit('--adapter is required with --engine hf')
V2 = a.format == 'v2'; CB = C2.load_codebook() if V2 else None
if V2 and a.max_new == 96: a.max_new = 16
if a.agent in ('observe', 'act') and not V2: sys.exit('observe/act are only supported with --format v2')
full = {}
for f in sorted(glob.glob(os.path.join(a.data_dir, f'{a.agent}_S?.jsonl'))):
    for l in open(f, encoding='utf-8'):
        r = json.loads(l); full[r['id']] = r
rows = []
for f in sorted(glob.glob(os.path.join(a.compact_dir, f'{a.agent}_S?.jsonl'))):
    rows += [json.loads(l) for l in open(f, encoding='utf-8')]
rows = [r for r in rows if r['split'] == 'test']
if a.n: rows = rows[::max(1, len(rows) // a.n)][:a.n]
nd = a.novelty_dir or os.path.join(os.path.dirname(os.path.abspath(a.compact_dir)), 'data_compact2'); novelty = {}
for f in sorted(glob.glob(os.path.join(nd, f'{a.agent}_S?.jsonl'))):
    for l in open(f, encoding='utf-8'):
        r = json.loads(l); novelty[r['id']] = r['input_seen_in_train']
schema = json.load(open(os.path.join(a.schema_dir, SCHEMA[a.agent] + '.schema.json')))
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = 'left'; tok.pad_token = tok.pad_token or '<|reserved_special_token_0|>'
if a.engine == 'llama':
    import llama_engine as LE
    eng = LE.LlamaEngine({a.agent: a.url.rstrip('/')}, tok); a.bsz = 1
    if not eng.health(a.agent): sys.exit(f'no llama-server answering at {a.url}')
else:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    import transformers
    _v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map={'': 0}, **{'dtype' if _v >= (4, 56) else 'torch_dtype': torch.bfloat16})
    model = PeftModel.from_pretrained(model, a.adapter).eval()
stop = [tok.eos_token_id, tok.convert_tokens_to_ids('<|eot_id|>')]
seen_n = collections.Counter(); seen_ok = collections.Counter(); kind_n = collections.Counter(); kind_ok = collections.Counter(); cat5 = 0
n = raw_ok = decoded = ok_schema = exact = 0; gen_tokens = 0; secs = 0.0; key_hit, key_tot = {}, {}
dumpf = open(a.dump, 'w', encoding='utf-8') if a.dump else None
for i in range(0, len(rows), a.bsz):
    b = rows[i:i + a.bsz]
    if a.engine == 'llama':
        txt, dt, ng, _ = eng.gen(a.agent, b[0]['messages'][:2], max_new=a.max_new); secs += dt; gen_tokens += ng; outs = [(b[0], txt)]
    else:
        enc = tok([tok.apply_chat_template(r['messages'][:2], tokenize=False, add_generation_prompt=True) for r in b], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        kw = {'prompt_lookup_num_tokens': a.prompt_lookup} if a.prompt_lookup and a.bsz == 1 else {}
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad(): out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, eos_token_id=stop, pad_token_id=tok.pad_token_id, **kw)
        torch.cuda.synchronize(); secs += time.time() - t; outs = []
        for r, o in zip(b, out):
            gen = o[enc['input_ids'].shape[1]:]; gen_tokens += int((gen != tok.pad_token_id).sum()); outs.append((r, tok.decode(gen, skip_special_tokens=True).strip()))
    for r, txt in outs:
        n += 1; raw_ok += txt == r['messages'][2]['content']
        g = full[r['id']]; ref = json.loads(g['messages'][2]['content']); inp = json.loads(g['messages'][1]['content'])
        try:
            if V2 and a.agent == 'observe': pred = C2.decode_observe2(txt, inp)
            elif V2 and a.agent == 'act':
                pred = C2.decode_act2(txt, inp)
                if a.guard: import act_guard as G; pred = G.apply(pred, inp)
            elif V2: pred = C2.decode_orient2(txt, inp, CB) if a.agent == 'orient' else C2.decode_decide2(txt, inp['context'], CB, ref['policy_id'])
            else: pred = K.decode_orient(txt, inp) if a.agent == 'orient' else K.decode_decide(txt, inp['context'], ref['policy_id'])
        except Exception as e:
            if dumpf: dumpf.write(json.dumps({'id': r['id'], 'error': 'decode_failed', 'raw': txt[:300]}) + '\n')
            continue
        decoded += 1
        try: validate(pred, schema); ok_schema += 1
        except ValidationError: pass
        if a.agent == 'observe': cat5 += (pred['load_level'], pred['qos_status']['severity'], pred['trend']['qos'], pred['trend']['load'], pred['anomalies']) == (ref['load_level'], ref['qos_status']['severity'], ref['trend']['qos'], ref['trend']['load'], ref['anomalies'])   # cat5
        kd = str(r.get('kind', '-')); kind_n[kd] += 1; kind_ok[kd] += pred == ref
        exact += pred == ref; sk = 'seen' if novelty.get(r['id'], r.get('input_seen_in_train', True)) else 'unseen'; seen_n[sk] += 1; seen_ok[sk] += pred == ref
        for k, v in ref.items(): key_tot[k] = key_tot.get(k, 0) + 1; key_hit[k] = key_hit.get(k, 0) + (pred.get(k) == v)
        if dumpf and pred != ref: dumpf.write(json.dumps({'id': r['id'], 'kind': r.get('kind'), 'raw': txt, 'expected': r['messages'][2]['content'], 'diff': {k: {'ref': ref[k], 'pred': pred.get(k)} for k in ref if pred.get(k) != ref[k]}}) + '\n')
    print(f'  scored {n}/{len(rows)}', end='\r', flush=True)
print(f'\n{a.agent} (compact): n={n}  decodable {decoded/n:.1%}  schema-valid {ok_schema/n:.1%}  exact match (full JSON after expansion) {exact/n:.1%}  raw compact string identical {raw_ok/n:.1%}')
print(f'  engine: {a.engine}  format: {a.format}' + ('  ACT GUARD ON' if a.guard else ''))
if a.agent == 'observe': print(f'  ALL 5 categorical decisions correct (load, severity, trend.qos, trend.load, anomalies): {cat5/n:.1%}   <- compare with the code version: 97.9 %.  (full-JSON exact match is lower for both because the numeric trend deltas come from rounded inputs)')
if novelty or V2: print('  exact match by input novelty (same test records for v1 and v2): ' + ', '.join(f'{k} {seen_ok[k]}/{seen_n[k]} = {seen_ok[k]/max(1,seen_n[k]):.1%}' for k in ('seen', 'unseen')) + '   (seen = this exact prompt occurs in train/val)')
if set(kind_n) != {'-'}:
    print('  exact match by record kind:')
    for k in sorted(kind_n): print(f'    {k:<34} {kind_ok[k]}/{kind_n[k]} = {kind_ok[k]/kind_n[k]:.1%}')
print(f'  generated tokens per example {gen_tokens/n:.1f}   seconds per example {secs/n:.2f}  (batch size {a.bsz})')
for k in sorted(key_tot, key=lambda k: key_hit[k] / key_tot[k]): print(f'  field {k:<28} {key_hit[k]/key_tot[k]:.1%}')
if dumpf: dumpf.close()
