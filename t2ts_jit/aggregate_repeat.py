#!/usr/bin/env python
"""Aggregate the 3-seed repeat runs for each best-FID checkpoint into mean±std.

Reads best_fid_ckpts.tsv (method,source,length,run_dir,ckpt,mode,orig_fid) and,
for each cell, the per-seed CSVs under repeat_best_fid/<cell>/seed{100,200,300}.csv.
Extracts the (ckpt, best-mode) row from each seed, computes mean/std over the 3
seeds for FID/CTTP/JFTSD, and writes evalaligner_best_ckpt_3seed.md (+ .csv).
"""
import csv, os, statistics as st
JIT="./t2ts_jit"
SEEDS=[100,200,300]
SOURCES=["ETTh1s","ETTm1","electricity","traffic"]; LENGTHS=[24,48,96]
METHODS=["v7","bridge","verbalts","t2s","diffusets","text2motion"]

cells={}
for r in csv.DictReader(open(os.path.join(JIT,"best_fid_ckpts.tsv")), delimiter="\t"):
    cells[(r["method"],r["source"],r["length"])]=(r["ckpt"],r["mode"],float(r["orig_fid"]))

def seed_val(cell, ckpt, mode, seed):
    p=os.path.join(JIT,"repeat_best_fid",cell,f"seed{seed}.csv")
    if not os.path.isfile(p): return None
    for r in csv.DictReader(open(p)):
        if r["ckpt"]==ckpt and r["mode"]==mode:
            return float(r["fid"]),float(r["cttp"]),float(r["jftsd"])
    return None

rows=[]; missing=[]
for m in METHODS:
    for s in SOURCES:
        for l in LENGTHS:
            ckpt,mode,ofid=cells[(m,s,str(l))]
            cell=f"{m}_{s}_L{l}"
            vals=[seed_val(cell,ckpt,mode,sd) for sd in SEEDS]
            vals=[v for v in vals if v is not None]
            if len(vals)<3: missing.append(f"{cell} ({len(vals)}/3)")
            if not vals: continue
            fids=[v[0] for v in vals]; ctt=[v[1] for v in vals]; jft=[v[2] for v in vals]
            def ms(x): return (sum(x)/len(x), (st.stdev(x) if len(x)>1 else 0.0))
            fm,fs=ms(fids); cm,cs=ms(ctt); jm,js=ms(jft)
            rows.append(dict(method=m,source=s,length=l,ckpt=ckpt,mode=mode,n=len(vals),
                fid_mean=fm,fid_std=fs,cttp_mean=cm,cttp_std=cs,jftsd_mean=jm,jftsd_std=js))

with open(os.path.join(JIT,"evalaligner_best_ckpt_3seed.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

L=["# Best-FID checkpoint — 3-seed re-sampling (mean ± std)\n",
   "For each (method, dataset) the checkpoint with the best FID under EvalAligner was",
   "re-sampled 3× (metrics-seeds 100/200/300). FID↓ / CTTP↑ / JFTSD↓.\n"]
def fmt(m,s): return f"{m:.4f}±{s:.4f}"
for s in SOURCES:
    for l in LENGTHS:
        L.append(f"\n## {s} L{l}\n")
        L.append("| method | FID↓ | CTTP↑ | JFTSD↓ | ckpt·mode |")
        L.append("|---|---|---|---|---|")
        for m in METHODS:
            r=next((x for x in rows if x["method"]==m and x["source"]==s and x["length"]==l),None)
            if r: L.append(f"| {m} | {fmt(r['fid_mean'],r['fid_std'])} | {fmt(r['cttp_mean'],r['cttp_std'])} | {fmt(r['jftsd_mean'],r['jftsd_std'])} | {r['ckpt'].replace('.pt','')}·{r['mode']} |")
            else: L.append(f"| {m} | — | — | — | missing |")
open(os.path.join(JIT,"evalaligner_best_ckpt_3seed.md"),"w").write("\n".join(L)+"\n")
print(f"wrote evalaligner_best_ckpt_3seed.md (+.csv), {len(rows)} cells")
if missing: print("INCOMPLETE:", ", ".join(missing))
