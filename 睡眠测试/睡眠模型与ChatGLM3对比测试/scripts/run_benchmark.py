#!/usr/bin/env python3
import argparse, json, time, urllib.request
from pathlib import Path

def post(url, payload):
    req=urllib.request.Request(url, data=json.dumps(payload,ensure_ascii=False).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r: return json.load(r)

ap=argparse.ArgumentParser()
ap.add_argument('--benchmark',default='benchmark.jsonl'); ap.add_argument('--output',required=True)
ap.add_argument('--url',default='http://127.0.0.1:8080/v1/chat/completions'); ap.add_argument('--model',default='chatglm3-6b-q4_k_m')
ap.add_argument('--limit',type=int); ap.add_argument('--seed',type=int,default=42)
ap.add_argument('--ids',help='comma-separated case IDs')
a=ap.parse_args(); rows=[json.loads(x) for x in Path(a.benchmark).read_text().splitlines() if x.strip()]
if a.ids:
    wanted=set(a.ids.split(',')); rows=[r for r in rows if r['id'] in wanted]
if a.limit: rows=rows[:a.limit]
out=[]
for item in rows:
    history=[]; answers=[]; t0=time.perf_counter(); usage={'completion_tokens':0,'prompt_tokens':0}
    for prompt in item['turns']:
        history.append({'role':'user','content':prompt}); q0=time.perf_counter()
        res=post(a.url,{'model':a.model,'messages':history,'temperature':0.2,'top_p':0.8,'max_tokens':256,'seed':a.seed,'stream':False})
        ans=res['choices'][0]['message']['content']; dt=time.perf_counter()-q0
        history.append({'role':'assistant','content':ans}); answers.append({'prompt':prompt,'response':ans,'latency_s':dt})
        for k in usage: usage[k]+=res.get('usage',{}).get(k,0)
    elapsed=time.perf_counter()-t0; ct=usage['completion_tokens']
    out.append({**item,'model':a.model,'generation':{'temperature':0.2,'top_p':0.8,'max_new_tokens':256,'seed':a.seed},'answers':answers,'latency_s':elapsed,'prompt_tokens':usage['prompt_tokens'],'completion_tokens':ct,'tokens_per_s':ct/elapsed if elapsed else None,'execution_status':'executed'})
    print(item['id'],round(elapsed,2),round(ct/elapsed,2) if elapsed else None,flush=True)
Path(a.output).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n')
