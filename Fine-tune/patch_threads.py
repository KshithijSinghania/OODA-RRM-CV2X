p = 'make_decide_coverage.py'; s = open(p, encoding='utf-8').read()
old = "    import build_decide as B_; B = B_\n"
new = ("    import build_decide as B_; B = B_\n"
       "    try:                                    # one CPLEX thread per worker: N workers x all cores oversubscribes the CPU\n"
       "        from docplex.mp import model as _M; _o = _M.Model.__init__\n"
       "        def _p(self, *x, **k): _o(self, *x, **k); self.parameters.threads = 1\n"
       "        _M.Model.__init__ = _p\n"
       "    except Exception as e: print('could not limit CPLEX threads:', e)\n")
assert s.count(old) == 1 and 'limit CPLEX threads' not in s
s = s.replace(old, new)
old2 = "    t0 = time.time(); res = pool.map(_solve, jobs);"
assert s.count(old2) == 1
s = s.replace(old2, "    print(f'checking {len(jobs)} existing records with the teacher ...', flush=True)\n" + old2)
open(p, 'w', encoding='utf-8').write(s); print('patched')
