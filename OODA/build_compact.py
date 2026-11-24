"""Write data_compact/{orient,decide}_S*.jsonl: same records and splits as data/, but the assistant target is the compact one-line format
(compact.py) and the system prompt describes it. Fine-tune with train_lora.py --data_dir data_compact (warm start from the JSON adapters)."""
import json, glob, os, collections
import compact as K
os.makedirs('data_compact', exist_ok=True); cnt = collections.Counter()
for ag, enc, sysmsg in (('orient', K.encode_orient, K.ORIENT_COMPACT_SYSTEM), ('decide', K.encode_decide, K.DECIDE_COMPACT_SYSTEM)):
    for f in sorted(glob.glob(f'data/{ag}_S?.jsonl')):
        out = []
        for l in open(f, encoding='utf-8'):
            r = json.loads(l); gold = json.loads(r['messages'][2]['content'])
            r['messages'] = [{'role': 'system', 'content': sysmsg}, r['messages'][1], {'role': 'assistant', 'content': enc(gold)}]
            out.append(json.dumps(r, separators=(',', ':')))
            cnt[(ag, r['split'])] += 1
        open(f'data_compact/{os.path.basename(f)}', 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print(dict(cnt))
