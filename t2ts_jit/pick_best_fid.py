#!/usr/bin/env python
"""For each (method, source, length) find the checkpoint+mode with the best (min)
FID from eval_evalaligner_metrics.csv. Emit a TSV driving the repeat-sampling runs.
Columns: method  source  length  run_dir  ckpt  mode  orig_fid
"""
import os
import csv, glob, os
JIT = os.path.dirname(os.path.abspath(__file__))
SOURCES=["ETTh1s","ETTm1","electricity","traffic"]; LENGTHS=[24,48,96]
METHODS=["v7","bridge","verbalts","t2s","diffusets","text2motion"]
V7PM={("ETTm1",24):"0.8",("ETTm1",96):"0.0",("traffic",24):"-0.8",("traffic",48):"-0.5",("traffic",96):"-0.8"}
def v7imgd(s,l):
    m,sd={24:(6,4),48:(8,6),96:(10,10)}[l]
    if s=="traffic" and l==24: m,sd=8,8
    return m,sd
def rundir(method,s,l):
    if method=="bridge": g=f"{JIT}/results_jit_cond_v7_bridge_{s}_L{l}_seed0"
    elif method=="verbalts": g=f"{JIT}/results_jit_cond_v7_verbalts_arch_{s}_L{l}"
    elif method=="diffusets": g=f"{JIT}/results_diffusets_baseline_{s}_L{l}_*"
    elif method=="t2s": g=f"{JIT}/results_t2s_baseline_{s}_L{l}_*"
    elif method=="text2motion": g=f"{JIT}/results_text2motion_baseline_{s}_L{l}_*"
    else:
        m,sd=v7imgd(s,l); pm=V7PM.get((s,l),"0.5")
        g=f"{JIT}/results_jit_cond_v7_{s}_L{l}_*pm{pm}_ps1.2_img{m}_d{sd}_bs512_lr1e-4"
    hits=[d for d in glob.glob(g) if "ablation" not in d and "dw0" not in d]
    return hits[0] if hits else None
rows=[]
for method in METHODS:
    for s in SOURCES:
        for l in LENGTHS:
            d=rundir(method,s,l); c=os.path.join(d,"eval_evalaligner_metrics.csv")
            best=None
            for r in csv.DictReader(open(c)):
                try: fid=float(r["fid"])
                except: continue
                if fid!=fid: continue
                if best is None or fid<best[0]: best=(fid,r["ckpt"],r["mode"])
            rows.append((method,s,str(l),d,best[1],best[2],f"{best[0]:.6f}"))
out=os.path.join(JIT,"best_fid_ckpts.tsv")
with open(out,"w") as f:
    f.write("method\tsource\tlength\trun_dir\tckpt\tmode\torig_fid\n")
    for r in rows: f.write("\t".join(r)+"\n")
print(f"wrote {out} ({len(rows)} cells)")
for r in rows: print("  ",r[0],r[1],r[2],"->",r[4],r[5],"fid="+r[6])
