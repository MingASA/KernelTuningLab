"""CPU JSON protocol fixture, explicitly NOT an LLM or trained model.

Use --once for stdin/stdout; HTTP mode is a bounded local contract fixture.
Real model providers implement the same decision object and identity metadata.
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler,HTTPServer


def decide(task):
    parameters=task['space']['parameters'];history=len(task['history'])
    if not parameters:return {'partial':{},'rationale':'protocol fixture, not a real LLM'}
    name=next(iter(parameters));description=parameters[name];values=description.get('values')
    if values:value=values[history%len(values)]
    else:value=description['default']
    return {'partial':{name:value},'rationale':'CPU protocol fixture; no model inference, training, or oracle'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true');p.add_argument('--port',type=int,default=0)
    p.add_argument('--requests',type=int,default=8);p.add_argument('--wall-seconds',type=float,default=30);a=p.parse_args()
    if a.once:
        task=json.load(sys.stdin);print(json.dumps(decide(task)));raise SystemExit(0)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n=int(self.headers.get('Content-Length','0'))
            if n>65536:self.send_error(413);return
            task=json.loads(self.rfile.read(n));blob=json.dumps(decide(task)).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(blob)
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',a.port),Handler);server.timeout=.5
    print(json.dumps({'port':server.server_port,'evidence':'protocol_fixture_not_LLM','max_requests':a.requests,'max_wall_s':a.wall_seconds}),flush=True)
    deadline=time.monotonic()+a.wall_seconds
    # handle_request may time out without a request; the wall bound still holds.
    while a.requests>0 and time.monotonic()<deadline:
        server.handle_request();a.requests-=1
    server.server_close()
