from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code'))
from src.core import read_jsonl
from src.study import h3_outcomes
from src.uf_trial import baselines

REPS = 10000
SEED = 261
def frame_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def interval(frame, cols, cluster='prompt_id'):
    """Row-weighted mean; resample whole prompt clusters, including all observed modes."""
    totals = frame.groupby(cluster,sort=True)[cols].sum().to_numpy(float)
    counts = frame.groupby(cluster,sort=True).size().to_numpy()
    m = len(counts)
    if not m:
        return [(None,None,None)]*len(cols)
    means = totals.sum(axis=0)/counts.sum()
    if m==1:
        return [(float(x),None,None) for x in means]
    rng = np.random.default_rng(SEED)
    draws=[]
    for start in range(0,REPS,250):
        indices=rng.integers(m,size=(min(250,REPS-start),m))
        draws.append(totals[indices].sum(axis=1)/counts[indices].sum(axis=1)[:,None])
    low,high=np.quantile(np.concatenate(draws),[.025,.975],axis=0)
    return list(zip(means,low,high))


def estimates(frame, cols, **labels):
    return [dict(**labels,metric=k,prompts=frame.prompt_id.nunique(),observations=len(frame),
                 mean=v,ci_low=lo,ci_high=hi) for k,(v,lo,hi) in zip(cols,interval(frame,cols))]



def replay():
    inputs = ROOT/'results'
    target = ROOT/'reproduced'
    target.mkdir(exist_ok=True)
    meta = json.loads((inputs/'manifest.json').read_text())
    mode_ids = meta['scope']
    state = meta['state']
    configs = meta['configs']
    prompts = read_jsonl(inputs/'frozen_inputs.jsonl')
    by_id = {p['prompt_id']:p for p in prompts}
    dec = pd.read_csv(inputs/'latest_decisions.csv',dtype={'candidate_a':str,'candidate_b':str})
    decisions = dec.to_dict('records')
    pilot_ids = set(dec.loc[dec.origin=='completed_recovery_pilot','prompt_id'])
    assert len(pilot_ids)==25
    att = pd.read_csv(inputs/'recorded_attempts.csv')
    stats=[]
    for cid in mode_ids:
        a=att[att.config_id==cid]; d=dec[dec.config_id==cid]
        last=d.merge(a[['request_id','attempt','strict_label']],on=['request_id','attempt'],validate='one_to_one')
        ok=last[last.status=='ok']
        stats.append(dict(config_id=cid,recorded_attempts=len(a),recorded_attempt1=sum(a.attempt==1),
            valid_latest=len(ok),invalid_latest=sum(d.status=='invalid_verdict'),truncated_latest=sum(d.status=='answer_truncated'),
            relaxed_valid_but_fails_strict_format=int((ok.strict_label!=ok.verdict).sum()),
            recorded_forced_closes=int(a.forced_close.fillna(False).sum()),recorded_generation_hours=a.allocated_seconds.sum()/3600))
    frame_csv(stats,target/'judge_diagnostics.csv')
    # Both accepted presentations have verdicts in ORIGINAL identity space.
    matches=dec[dec.status=='ok'].pivot(index=['config_id','prompt_id','match_name'],columns='order',values='verdict').reindex(columns=['AB','BA']).dropna().reset_index()
    matches['order_disagreement']=matches.AB!=matches.BA
    matches['displayed_first_rate']=((matches.AB=='A').astype(float)+(matches.BA=='B').astype(float))/2
    matches.to_csv(target/'order_pairs.csv',index=False)
    match_rows=[]
    for cid in mode_ids:
        match_rows+=estimates(matches[matches.config_id==cid],['order_disagreement','displayed_first_rate'],config_id=cid)
    frame_csv(match_rows,target/'order_diagnostics.csv')

    o=pd.DataFrame(h3_outcomes(prompts,configs,decisions))
    base=pd.DataFrame(baselines(prompts))
    o=o.merge(base,on=['prompt_id','n'],validate='many_to_one')
    o['pilot']=o.prompt_id.isin(pilot_ids)
    answers={pid:{a['answer_id']:a for a in p['answers']} for pid,p in by_id.items()}
    def markdown(pid,aid): return int(answers[pid][aid]['markdown'])
    o['gold_markdown']=[markdown(r.prompt_id,r.gold_id) for r in o.itertuples()]
    o['selected_markdown']=[np.nan if r.missing else markdown(r.prompt_id,r.selected_id) for r in o.itertuples()]
    o['best_quality_selected']=np.where(o.missing,np.nan,(o.regret==0).astype(float))
    o['quality_gain_over_random']=o.selected_quality-o.random_expected_quality
    assert np.allclose(o.loc[~o.missing,'format_drift'],(o.selected_markdown-o.gold_markdown).dropna())
    o.to_csv(target/'prompt_outcomes.csv',index=False)
    base.to_csv(target/'baselines.csv',index=False)
    n4=o[(o.n==4)&~o.missing]
    complete_ids={cid:set(n4[n4.config_id==cid].prompt_id) for cid in mode_ids}
    common=set.intersection(*complete_ids.values())
    matched=o.merge(n4[['config_id','prompt_id']],on=['config_id','prompt_id'],validate='many_to_one')
    cohorts={'all_500':set(by_id),'new_475':set(by_id)-pilot_ids,'pilot_25':pilot_ids,
             'no_gpt4_candidates':{pid for pid,p in by_id.items() if p['without_gpt4']}}
    cohorts['new_no_gpt4']=cohorts['new_475'] & cohorts['no_gpt4_candidates']
    metrics=['format_drift','regret','selected_quality','best_quality_selected','selected_markdown','gold_markdown','quality_gain_over_random']
    pooled=[]; summaries=[]; pairs=[]
    for label,ids in cohorts.items():
        group=n4[n4.prompt_id.isin(ids)]
        for denominator,f in [('available',group),('common_six',group[group.prompt_id.isin(common)])]:
            pooled+=estimates(f,metrics,cohort=label,denominator=denominator,eligible_prompts=len(ids))
        for cid in mode_ids:
            f=matched[(matched.config_id==cid)&matched.prompt_id.isin(ids)]
            for n in (1,2,4):
                sample=f[f.n==n]
                summaries+=estimates(sample,metrics,cohort=label,config_id=cid,n=n,eligible_prompts=len(ids))
        # Primary mode comparisons and exploratory adjacent-size contrasts use the same prompts on both sides.
        contrasts=[(mode_ids[i+1],mode_ids[i],'thinking_on_minus_off') for i in (0,2,4)]
        contrasts += [(mode_ids[i+2],mode_ids[i],'larger_minus_smaller') for i in range(4)]
        for high,low,kind in contrasts:
            high_rows=group[group.config_id==high].set_index('prompt_id')[metrics]
            low_rows=group[group.config_id==low].set_index('prompt_id')[metrics]
            diff=(high_rows-low_rows).dropna().reset_index()
            pairs+=estimates(diff,metrics[:4],cohort=label,contrast=kind,left=high,right=low)
    pool=pd.DataFrame(pooled); summary=pd.DataFrame(summaries)
    pool.to_csv(target/'pooled_h3.csv',index=False)
    summary.to_csv(target/'per_mode.csv',index=False)
    frame_csv(pairs,target/'paired_comparisons.csv')
    sources=[]
    for source,group in n4.groupby('source'):
        sources+=estimates(group,metrics[:4],source=source)
    frame_csv(sources,target/'by_source_exploratory.csv')

    fig,axes=plt.subplots(1,3,figsize=(13,4))
    titles=['Selected proxy quality (1–5)','Regret (rating points)','Markdown drift (percentage points)']
    for ax,metric,title in zip(axes,['selected_quality','regret','format_drift'],titles):
        for i,cid in enumerate(mode_ids):
            r=summary[(summary.cohort=='all_500')&(summary.config_id==cid)&(summary.metric==metric)]
            ax.plot(r.n,r['mean']*(100 if metric=='format_drift' else 1),marker='o',ls='-' if i%2 else '--',
                    color=['#2056a8','#247353','#ab5725'][i//2],label=cid.replace('qwen3-',''))
        ax.set(title=title,xlabel='Number of candidates',xticks=[1,2,4])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'UltraFeedback · {state.lower()} six-mode results · complete brackets within each mode')
    fig.tight_layout();fig.savefig(target/'selection_curves.png',dpi=180);plt.close(fig)
    headline=pool[(pool.metric=='format_drift')&(pool.denominator=='available')].set_index('cohort')
    fig,ax=plt.subplots(figsize=(8,3.7))
    for i,(label,r) in enumerate(headline.iterrows()):
        if pd.notna(r.ci_low):
            ax.plot([100*r.ci_low,100*r.ci_high],[i,i],color='#52687a')
        ax.plot(100*r['mean'],i,'o',color='#2056a8')
    ax.axvline(0,ls='--',color='gray');ax.set(yticks=range(len(headline)),yticklabels=headline.index,
        xlabel='Best-of-4 Markdown drift (percentage points), 95% prompt-cluster interval',title=f'{state}: six-mode pooled estimates')
    fig.tight_layout();fig.savefig(target/'pooled_drift.png',dpi=180);plt.close(fig)

    for name in ('prompt_outcomes','pooled_h3','per_mode','paired_comparisons','by_source_exploratory','order_diagnostics','judge_diagnostics','baselines'):
        expected=pd.read_csv(inputs/(name+'.csv'))
        observed=pd.read_csv(target/(name+'.csv'))
        pd.testing.assert_frame_equal(expected, observed, check_exact=False, rtol=1e-10, atol=1e-12)
    print('Verified 9,000 prompt outcomes and all eight published analytical tables. See reproduced/ for regenerated tables and figures.')
    return headline[['prompts','observations','mean','ci_low','ci_high']]

if __name__ == '__main__':
    print(replay())
