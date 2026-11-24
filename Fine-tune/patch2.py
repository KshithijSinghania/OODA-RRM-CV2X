p = 'make_decide_coverage.py'; s = open(p, encoding='utf-8').read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old; s = s.replace(old, new)
rep("def _init(ooda):", "def _init(ooda, threads=1):")
rep("    try:                                    # one CPLEX thread", "    try:\n      if threads > 0:                       # one CPLEX thread")
rep("        from docplex.mp import model as _M; _o = _M.Model.__init__\n        def _p(self, *x, **k): _o(self, *x, **k); self.parameters.threads = 1\n        _M.Model.__init__ = _p\n",
    "        from docplex.mp import model as _M; _o = _M.Model.__init__\n        def _p(self, *x, **k): _o(self, *x, **k); self.parameters.threads = threads\n        _M.Model.__init__ = _p\n")
rep("pool = mp.Pool(a.procs, _init, (ooda,))", "pool = mp.Pool(a.procs, _init, (ooda, a.threads))")
rep("ap.add_argument('--seed', type=int, default=7); a = ap.parse_args()",
    "ap.add_argument('--seed', type=int, default=7); ap.add_argument('--threads', type=int, default=1, help='CPLEX threads per worker, 0 = do not limit'); ap.add_argument('--min_faith', type=float, default=0.97); a = ap.parse_args()")
rep("        ok += code == r['messages'][2]['content']; bad += code != r['messages'][2]['content']\n",
    "        same = code == r['messages'][2]['content']; ok += same; bad += not same\n"
    "        if not same and len(diffs) < 14:\n"
    "            g = r['messages'][2]['content']; diffs.append((r['messages'][1]['content'], g, code, [n for n, sl in (('PRB', slice(0, 6)), ('power', slice(6, 9)), ('strat', slice(9, 10)), ('MEC', slice(10, 12))) if g[sl] != code[sl]]))\n")
rep("cbx['mec'] = list(cb['mec']); ok = bad = fail = 0", "cbx['mec'] = list(cb['mec']); ok = bad = fail = 0; diffs = []")
rep("    if ok < 0.97 * len(rec): sys.exit(",
    "    for pr, g, c, f in diffs: print(f'  DIFF fields={f}\\n    prompt  : {pr}\\n    stored  : {g}\\n    teacher : {c}')\n"
    "    if ok < a.min_faith * len(rec): sys.exit(")
open(p, 'w', encoding='utf-8').write(s); print('patched2')
