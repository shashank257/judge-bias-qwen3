from pathlib import Path
import csv,numpy as np
P=Path(__file__).resolve().parent
rows=list(csv.DictReader((P/'reproduced/outcomes.csv').open()))
rr=[r for r in rows if r['n']=='4' and r['missing']=='False']
configs=sorted({r['config_id'] for r in rr})
common=set.intersection(*[{r['prompt_id'] for r in rr if r['config_id']==c} for c in configs])
out=[]
for cohort in ['all_available','common_all_six','without_gpt4_available']:
 r=[x for x in rr if (cohort!='common_all_six' or x['prompt_id'] in common) and (cohort!='without_gpt4_available' or x['without_gpt4']=='True')]
 ids=sorted({x['prompt_id'] for x in r});draws=np.random.default_rng(261).integers(0,len(ids),size=(10000,len(ids)))
 for metric in ['format_drift','regret','selected_quality']:
  sums=np.array([sum(float(x[metric]) for x in r if x['prompt_id']==p) for p in ids]);counts=np.array([sum(x['prompt_id']==p for x in r) for p in ids])
  lo,hi=np.quantile(sums[draws].sum(1)/counts[draws].sum(1),[.025,.975])
  out.append(dict(cohort=cohort,metric=metric,prompts=len(ids),configuration_prompt_rows=len(r),mean=sums.sum()/counts.sum(),ci_low=lo,ci_high=hi,bootstrap_replicates=10000,seed=261))
with (P/'reproduced/pooled.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print('Wrote reproduced/pooled.csv')
