import json,csv,collections,random,math,os
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]; OUT=P/'results'/'analysis';OUT.mkdir(parents=True, exist_ok=True)
import importlib.util
spec=importlib.util.spec_from_file_location('h',P/'src'/'run_judges.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
jobs,_=h.build_jobs(h.read_pairs(P/'data'/'mtbench_matched_31.csv')); byid={j['job_id']:j for j in jobs}; assert len(byid)==310
sources={
'0.6B OFF':['qwen3-0.6b_matched31_off_fp16.jsonl'],
'0.6B ON':['qwen3_06b_matched31_on_local.jsonl'],
'1.7B OFF':['qwen3-1.7b_matched31_off_fp16.jsonl'],
'1.7B ON':['qwen3-1.7b_matched31_on_temp06_top095_seeded_v2_fp16.jsonl'],
'4B OFF':['qwen3_4b_instruct_matched31_off.jsonl'],
'4B ON':['qwen3_4b_matched31_on_consolidated.jsonl'],
'8B OFF':['qwen3_8b_50_native_off.jsonl'],
'8B ON':['qwen3_8b_50_on_4096_ctx8192.jsonl','qwen3_8b_matched31_on_recovery.jsonl','qwen3-8b_matched31_on_nf4_sampled_recovery.jsonl']}
records=[]; audits=[]
for cfg,fs in sources.items():
 ok={}; counts=[]; conflicts=[]
 for fn in fs:
  rr=[json.loads(line) for line in open(P/'results'/'logs'/fn) if line.strip()]; counts.append((fn,len(rr),sum(x.get('status')=='ok' for x in rr)))
  for r in rr:
   jid=r.get('job_id'); w=r.get('judge_winner_original')
   if jid not in byid or r.get('status')!='ok' or w not in ('A','B'): continue
   if jid in ok:
    if ok[jid]['judge_winner_original']!=w: conflicts.append(jid)
    continue
   ok[jid]=r
 for jid,r in ok.items():
  j=byid[jid]
  assert all(str(j[k])==str(r[k]) for k in ('pair_id','condition','order','replicate'))
  records.append({**{k:j[k] for k in ('job_id','pair_id','question_id','human_winner','condition','order','replicate','displayed_first')},'config':cfg,'winner':r['judge_winner_original'],'correct':int(r['judge_winner_original']==j['human_winner']),'selected_first':int(r['judge_winner_original']==j['displayed_first'])})
 audits.append({'config':cfg,'valid_unique':len(ok),'missing':310-len(ok),'overlap_conflicts':len(conflicts),'source_counts':counts})
df=pd.DataFrame(records);df.to_csv(OUT/'judgments_valid.csv',index=False)
print('audit',audits)
# Per-pair features: repeat noise, plain/order and markdown/order; where suitable paired replicate-0 comparisons
PAIR_METRICS=['human_agreement','first_preference','repeat_flip','plain_order_flip','format_flip_any','markdown_advantage']
rows=[]
for (cfg,pid),group in df.groupby(['config','pair_id']):
 g={(r.condition,r.order,int(r.replicate)):r for r in group.itertuples()}
 def r(c,o,n=0):return g.get((c,o,n))
 plain='A_plain__B_plain';both='A_markdown__B_markdown'; a='A_markdown__B_plain';b='A_plain__B_markdown'
 ret={'config':cfg,'pair_id':pid,'human_agreement':np.mean([x.correct for x in group.itertuples()]),'first_preference':np.mean([x.selected_first for x in group.itertuples()])}
 rep=[int(r(plain,o,0).winner!=r(plain,o,1).winner) for o in ['AB','BA'] if r(plain,o,0) is not None and r(plain,o,1) is not None]
 ret['repeat_flip']=float(np.mean(rep)) if rep else np.nan
 if r(plain,'AB') is not None and r(plain,'BA') is not None:ret['plain_order_flip']=int(r(plain,'AB').winner!=r(plain,'BA').winner)
 else:ret['plain_order_flip']=np.nan
 fmt=[];adv=[]
 for o in ['AB','BA']:
  p=r(plain,o);am=r(a,o);bm=r(b,o);bothm=r(both,o)
  if p is not None:
   for v in [am,bm,bothm]:
    if v is not None:fmt.append(int(p.winner!=v.winner))
  if am is not None:adv.append(int(am.winner=='A'))
  if bm is not None:adv.append(int(bm.winner=='B'))
 ret['format_flip_any']=float(np.mean(fmt)) if fmt else np.nan
 ret['markdown_advantage']=float(np.mean(adv)) if adv else np.nan
 rows.append(ret)
pair=pd.DataFrame(rows);pair.to_csv(OUT/'pair_metrics.csv',index=False)
# cluster bootstrap by pair; estimates pair-averaged for all metrics
rng=np.random.default_rng(261); summary=[]
for cfg,gg in pair.groupby('config'):
 for metric in PAIR_METRICS:
  x=gg[metric].dropna().to_numpy(float);n=len(x)
  if n:
   bs=x[rng.integers(0,n,size=(4000,n))].mean(axis=1)
   lo,hi=np.quantile(bs,[.025,.975]); val=np.mean(x)
  else:val=lo=hi=np.nan
  summary.append({'config':cfg,'metric':metric,'estimate':val,'lo95':lo,'hi95':hi,'pairs':n})
sumdf=pd.DataFrame(summary);sumdf.to_csv(OUT/'metric_summary.csv',index=False)
# pair matched ON minus OFF delta with paired boot
comparisons=[]
for size in ('0.6B','1.7B','4B','8B'):
 for m in PAIR_METRICS:
  a=pair[pair.config==size+' ON'][['pair_id',m]].rename(columns={m:'on'});b=pair[pair.config==size+' OFF'][['pair_id',m]].rename(columns={m:'off'});mm=a.merge(b,on='pair_id').dropna();n=len(mm)
  if n:
   d=(mm['on']-mm['off']).to_numpy(); bs=d[rng.integers(0,n,size=(4000,n))].mean(axis=1);lo,hi=np.quantile(bs,[.025,.975]);delta=d.mean()
  else:delta=lo=hi=np.nan
  comparisons.append({'size':size,'metric':m,'difference_on_minus_off':delta,'lo95':lo,'hi95':hi,'paired_pairs':n})
pd.DataFrame(comparisons).to_csv(OUT/'thinking_comparisons.csv',index=False)
with open(OUT/'audit.json','w') as f:json.dump(audits,f,indent=2)
print(sumdf.pivot(index='config',columns='metric',values='estimate').round(3).to_string())
print('saved',OUT)
