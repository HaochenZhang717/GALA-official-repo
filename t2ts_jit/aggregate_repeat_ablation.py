#!/usr/bin/env python
"""Aggregate ablation/compare 3-seed repeats into mean±std.
Reads best_fid_ablation.tsv + repeat_ablation/<cell>/seed{100,200,300}.csv.
Writes t2ts_jit/ablation_best_ckpt_3seed.csv (groups noalign/joint/df_n_arch)
   and t2ts_jit/compare_best_ckpt_3seed.csv (groups jepa/structure)
   and t2ts_jit/ablation_broken.txt (settings with unreproducible conditioning).
"""
import csv, os, statistics as st
JIT="./t2ts_jit"
SEEDS=[100,200,300]
def sv(cell, ckpt, mode, seed):
    p=os.path.join(JIT,"repeat_ablation",cell,f"seed{seed}.csv")
    if not os.path.isfile(p): return None
    for r in csv.DictReader(open(p)):
        if r["ckpt"]==ckpt and r["mode"]==mode:
            return float(r["fid"]),float(r["cttp"]),float(r["jftsd"])
    return None
def ms(x): return (sum(x)/len(x), (st.stdev(x) if len(x)>1 else 0.0))

abl=[]; cmp=[]; broken=[]; incomplete=[]
for r in csv.DictReader(open(os.path.join(JIT,"best_fid_ablation.tsv")), delimiter="\t"):
    if r["ok"]!="1":
        broken.append(f"{r['group']}/{r['setting']} ({r['source']}) — {r['note']} (maxCTTP={r['max_cttp']})"); continue
    cell=os.path.basename(r["run_dir"]); ckpt,mode=r["ckpt"],r["mode"]
    vals=[sv(cell,ckpt,mode,s) for s in SEEDS]; vals=[v for v in vals if v]
    if len(vals)<3: incomplete.append(f"{cell} ({len(vals)}/3)")
    if not vals: continue
    fm,fs=ms([v[0] for v in vals]); cm,cs=ms([v[1] for v in vals]); jm,js=ms([v[2] for v in vals])
    row=dict(group=r["group"],setting=r["setting"],source=r["source"],length="48",
             ckpt=ckpt,mode=mode,n=len(vals),fid_mean=fm,fid_std=fs,
             cttp_mean=cm,cttp_std=cs,jftsd_mean=jm,jftsd_std=js)
    (cmp if r["group"] in ("jepa","structure") else abl).append(row)

def dump(path, rows):
    if not rows: return
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
dump(os.path.join(JIT,"ablation_best_ckpt_3seed.csv"), abl)
dump(os.path.join(JIT,"compare_best_ckpt_3seed.csv"), cmp)
open(os.path.join(JIT,"ablation_broken.txt"),"w").write("\n".join(broken)+"\n" if broken else "")
print(f"ablation rows={len(abl)}  compare rows={len(cmp)}  broken={len(broken)}  incomplete={len(incomplete)}")
if broken: print("BROKEN:", *broken, sep="\n  ")
if incomplete: print("INCOMPLETE:", *incomplete, sep="\n  ")
