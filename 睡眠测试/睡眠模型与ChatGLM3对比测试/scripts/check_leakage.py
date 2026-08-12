#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path

def norm(s): return re.sub(r'\W+','',s.lower())
def grams(s,n=5):
    s=norm(s); return {s[i:i+n] for i in range(max(0,len(s)-n+1))}
def sim(a,b):
    a,b=grams(a),grams(b); return len(a&b)/len(a|b) if a|b else 0
ap=argparse.ArgumentParser(); ap.add_argument('--benchmark',default='benchmark.jsonl'); ap.add_argument('--train',required=True); ap.add_argument('--report',required=True); a=ap.parse_args()
bench=[json.loads(x) for x in Path(a.benchmark).read_text().splitlines() if x.strip()]
txt=Path(a.train).read_text()
try: raw=json.loads(txt)
except json.JSONDecodeError: raw=[json.loads(x) for x in txt.splitlines() if x.strip()]
def collect(x):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            if k in ('content','prompt','query','instruction','input') and isinstance(v,str): out.append(v)
            out+=collect(v)
    elif isinstance(x,list):
        for v in x: out+=collect(v)
    return out
train=collect(raw); findings=[]
for b in bench:
    for ti,t in enumerate(b['turns']):
        best=max(((sim(t,x),x) for x in train),default=(0,''))
        findings.append({'id':b['id'],'turn':ti+1,'max_5gram_jaccard':round(best[0],4),'nearest_train_text':best[1]})
report={'method':'normalized character 5-gram Jaccard against all extracted training prompt/content strings','threshold':0.65,'benchmark_turns':len(findings),'training_strings':len(train),'flagged':[x for x in findings if x['max_5gram_jaccard']>=0.65],'all':findings}
Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
