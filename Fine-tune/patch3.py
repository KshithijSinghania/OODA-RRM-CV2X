p = 'make_decide_coverage.py'; s = open(p, encoding='utf-8').read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old; s = s.replace(old, new)
rep("ap.add_argument('--min_faith', type=float, default=0.97); a = ap.parse_args()",
    "ap.add_argument('--min_faith', type=float, default=0.97)\n    ap.add_argument('--dense_n', type=int, default=0, help='sample EVERY n in 1..dense_n and EVERY vehicles-within-300m count 0..n (default 0 = the sparse lists)'); a = ap.parse_args()")
rep("    LOADS, QOS = ['low', 'moderate', 'high', 'very_high'],",
    "    if a.dense_n: NS = sorted(set(range(1, a.dense_n + 1)) | {x for x in NS if x > a.dense_n})\n    LOADS, QOS = ['low', 'moderate', 'high', 'very_high'],")
rep("for m in N300 if m <= n]", "for m in (range(n + 1) if n <= a.dense_n else [x for x in N300 if x <= n])]")
open(p, 'w', encoding='utf-8').write(s); print('patched3')
