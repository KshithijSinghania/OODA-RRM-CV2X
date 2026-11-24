"""llama.cpp (llama-server) inference engine for the OODA hops: one server per agent (merged + quantised GGUF), HTTP/JSON, no extra Python packages.

Why: the measured cost of one Orient/Decide hop on the HF + 4-bit + unmerged-LoRA path is about 0.2 s overhead + 0.0006 s per prompt token + 0.07 s per generated token
(fit_latency.py on your own traces). Decode dominates; llama.cpp on a merged Q4 model decodes several times faster on the same GPU.

The prompt is rendered with the SAME Hugging Face chat template as training/eval (so the text is identical); only the leading <|begin_of_text|> is removed because
llama-server adds BOS itself. check() asks the server to tokenise that exact string and compares with the HF token ids - run it once, it catches template/BOS mismatches.
API used (llama-server): POST /completion {prompt, n_predict, temperature, cache_prompt, stop} -> {content, tokens_predicted, tokens_evaluated, timings{...}};
POST /tokenize {content, add_special, parse_special} -> {tokens}; GET /health.   Not exercised against a real llama-server in the authoring environment: see README_INFER.md step 6."""
import json, time, urllib.request, urllib.error

def parse_urls(s):
    """'orient=http://127.0.0.1:8081,decide=http://127.0.0.1:8082' -> dict"""
    d = {}
    for part in (s or '').split(','):
        if part.strip():
            k, v = part.split('=', 1); d[k.strip()] = v.strip().rstrip('/')
    return d

def _post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode('utf-8'))

class LlamaEngine:
    def __init__(self, urls, tok, timeout=120, cache_prompt=True):
        self.urls = urls; self.tok = tok; self.timeout = timeout; self.cache_prompt = cache_prompt; self.warm = set(); self.checked = set(); self.last = {}

    def text(self, messages):
        t = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        bos = getattr(self.tok, 'bos_token', None) or '<|begin_of_text|>'
        return t, (t[len(bos):] if t.startswith(bos) else t)          # (full text as HF sees it, text to send: server adds BOS)

    def health(self, agent):
        try:
            with urllib.request.urlopen(self.urls[agent] + '/health', timeout=10) as r: return r.status == 200
        except Exception: return False

    def warmup(self, agent):
        if agent in self.warm: return
        _post(self.urls[agent] + '/completion', {'prompt': 'Hello', 'n_predict': 2, 'temperature': 0, 'cache_prompt': False}, self.timeout); self.warm.add(agent)

    def check(self, agent, messages):
        """Compare the server's tokenisation of the string we send with the HF token ids. Returns (ok, message)."""
        full, send = self.text(messages)
        ref = self.tok(full, add_special_tokens=False)['input_ids'] if callable(self.tok) else None
        try: got = _post(self.urls[agent] + '/tokenize', {'content': send, 'add_special': True, 'parse_special': True}, self.timeout)['tokens']
        except Exception as e: return False, f'could not call /tokenize on {self.urls[agent]}: {e}'
        got = [t['id'] if isinstance(t, dict) else t for t in got]
        if ref is None: return True, 'no HF tokenizer to compare with'
        if list(ref) == list(got): return True, f'tokenisation identical ({len(got)} tokens)'
        n = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), min(len(ref), len(got)))
        return False, f'TOKENISATION MISMATCH for {agent}: HF {len(ref)} tokens vs server {len(got)}; first difference at position {n} (HF {list(ref)[n:n+3]} vs server {list(got)[n:n+3]})'

    def gen(self, agent, messages, max_new=96):
        """-> (text, seconds, n_generated, n_prompt). Greedy decoding, same as the HF evaluation. Seconds = wall time of the HTTP call."""
        if agent not in self.urls: raise KeyError(f'no llama-server URL for agent {agent!r}; pass --llama_urls {agent}=http://127.0.0.1:PORT')
        self.warmup(agent)
        if agent not in self.checked:
            ok, msg = self.check(agent, messages); self.checked.add(agent); print(('  [engine] ' if ok else '  [engine] WARNING: ') + msg, flush=True)
        _, send = self.text(messages)
        t = time.time()
        r = _post(self.urls[agent] + '/completion', {'prompt': send, 'n_predict': max_new, 'temperature': 0.0, 'top_k': 1, 'cache_prompt': self.cache_prompt,
                                                      'stop': ['<|eot_id|>'], 'stream': False}, self.timeout)
        dt = time.time() - t; self.last = r
        return r['content'].strip(), dt, int(r.get('tokens_predicted', 0)), int(r.get('tokens_evaluated', 0))
