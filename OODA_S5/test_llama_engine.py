"""Contract test of llama_engine.py against an in-process stand-in for llama-server (request shape, BOS handling, tokenisation check, response parsing, error path).
Does NOT prove compatibility with a real llama-server; README_INFER.md step 6 does that on the laptop."""
import json, threading, http.server, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llama_engine as LE
BOS = '<|begin_of_text|>'
class StubTok:
    bos_token = BOS
    def apply_chat_template(self, m, tokenize=False, add_generation_prompt=True):
        return BOS + ''.join(f"<|start_header_id|>{x['role']}<|end_header_id|>\n\n{x['content']}<|eot_id|>" for x in m) + '<|start_header_id|>assistant<|end_header_id|>\n\n'
    def __call__(self, text, add_special_tokens=False):
        ids = [1 if w == BOS else 100 + (hash(w) % 50) * 0 + len(w) for w in text.replace(BOS, ' ' + BOS + ' ').split()]   # deterministic toy ids
        return {'input_ids': ids}
seen = []
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, o, code=200): b = json.dumps(o).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send({'status': 'ok'}) if self.path == '/health' else self._send({}, 404)
    def do_POST(self):
        p = json.loads(self.rfile.read(int(self.headers['Content-Length']))); seen.append((self.path, p))
        if self.path == '/tokenize':
            text = (BOS + ' ' if p.get('add_special') else '') + p['content']
            if getattr(H, 'break_tok', False): text += ' extra'
            self._send({'tokens': StubTok()(text)['input_ids']})
        elif self.path == '/completion':
            self._send({'content': ' base|0.4,0.4,0.2|20,70,500|tf|-|- ', 'tokens_predicted': 21, 'tokens_evaluated': 640, 'tokens_cached': 200, 'timings': {'prompt_ms': 300.0, 'predicted_ms': 400.0}})
srv = http.server.HTTPServer(('127.0.0.1', 0), H); port = srv.server_address[1]; threading.Thread(target=srv.serve_forever, daemon=True).start()
urls = LE.parse_urls(f'orient=http://127.0.0.1:{port}/, decide=http://127.0.0.1:{port}'); assert urls['orient'].endswith(str(port)), urls
eng = LE.LlamaEngine(urls, StubTok()); msgs = [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'u 1 2 3'}]
assert eng.health('orient')
txt, dt, n, p = eng.gen('orient', msgs, max_new=96)
assert txt == 'base|0.4,0.4,0.2|20,70,500|tf|-|-' and n == 21 and p == 640 and dt >= 0, (txt, n, p)
comp = [x for path, x in seen if path == '/completion' and x['prompt'] != 'Hello'][0]
assert not comp['prompt'].startswith(BOS), 'BOS must be stripped (server adds it)'
assert comp['temperature'] == 0.0 and comp['n_predict'] == 96 and comp['cache_prompt'] is True and '<|eot_id|>' in comp['stop'], comp
assert any(path == '/tokenize' and x['add_special'] and x['parse_special'] for path, x in seen)
H.break_tok = True; eng2 = LE.LlamaEngine(urls, StubTok()); ok, msg = eng2.check('orient', msgs); assert not ok and 'MISMATCH' in msg, msg
try: eng.gen('act', msgs); raise SystemExit('expected KeyError')
except KeyError as e: assert 'act' in str(e)
print('PASS llama_engine contract test:', msg[:90])
