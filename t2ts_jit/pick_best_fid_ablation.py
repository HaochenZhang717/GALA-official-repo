#!/usr/bin/env python
"""Pick best-FID ckpt/mode per ablation/compare setting from step-1 CSVs.
Flags settings whose max CTTP < 0.10 as broken-conditioning (ok=0) — e.g. noalign_lora
(stage-2-trained LoRA text encoder not saved in ckpt, so conditioning is unreproducible).
Writes t2ts_jit/best_fid_ablation.tsv:
  exp group setting source run_dir ckpt mode best_fid max_cttp ok extra
"""
import os
import csv, glob, os
JIT = os.path.dirname(os.path.abspath(__file__))
GROUP={"noalign":"noalign","joint":"joint","ft":"df_n_arch","jepa":"jepa","structure":"structure","tuneside":"tuneside","persample":"persample"}

def label(exp, extra, source):
    if exp=="noalign": return dict(kv(extra)).get("TEXT_VARIANT","?")
    if exp=="joint":   return "joint (single-stage)"
    if exp=="ft":
        d=dict(kv(extra)); return f"ft{d.get('FT_METHOD','?')}·dw{d.get('ALIGNER_DW','?')}"
    if exp=="tuneside": return "tune-"+dict(kv(extra)).get("TUNE_SIDE","?")
    return source  # jepa/structure grouped by source
def kv(extra):
    return [] if extra.strip()=="-" else [p.split("=",1) for p in extra.split()]

rows=[]
for r in csv.DictReader(open(os.path.join(JIT,"ablation_compare_settings.tsv")), delimiter="\t"):
    exp,source,extra=r["exp_type"],r["source"],r["extra_env"]
    d=r["run_dir"]; c=os.path.join(d,"eval_evalaligner_metrics.csv")
    if not os.path.isfile(c): rows.append((exp,GROUP[exp],label(exp,extra,source),source,d,"","","", "","0",extra,"NO_CSV")); continue
    best=None; maxc=-9
    for x in csv.DictReader(open(c)):
        try: fid=float(x["fid"]); cttp=float(x["cttp"])
        except: continue
        if fid!=fid: continue
        maxc=max(maxc,cttp)
        if best is None or fid<best[0]: best=(fid,x["ckpt"],x["mode"])
    ok = "1" if maxc>=0.10 else "0"
    note = "" if ok=="1" else "BROKEN_CONDITIONING(cttp~0)"
    rows.append((exp,GROUP[exp],label(exp,extra,source),source,d,best[1],best[2],f"{best[0]:.6f}",f"{maxc:.4f}",ok,extra,note))

out=os.path.join(JIT,"best_fid_ablation.tsv")
with open(out,"w") as f:
    f.write("exp\tgroup\tsetting\tsource\trun_dir\tckpt\tmode\tbest_fid\tmax_cttp\tok\textra\tnote\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+"\n")
print(f"wrote {out} ({len(rows)} settings)")
for r in rows: print(f"  [{r[9]}] {r[1]:10s} {r[2]:16s} {r[3]:11s} best={r[5]}·{r[6]} fid={r[7]} maxCTTP={r[8]} {r[11]}")
nbad=sum(1 for r in rows if r[9]=="0")
print(f"\nOK={len(rows)-nbad}/{len(rows)}  BROKEN={nbad}")
