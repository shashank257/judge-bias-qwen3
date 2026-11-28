# UltraFeedback: six-mode CPU analysis. Run as one cell after the full notebook's setup
# or use the standalone Colab notebook. No inference or model downloads.
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MODE_IDS = [f'qwen3-{size}-{mode}' for size in ('0.6b','1.7b','4b') for mode in ('off','on')]
CHUNKS = {'0.6b':125, '1.7b':50, '4b':25}
REPS = 10000
SEED = 261


def locate():
    if 'run' in globals() and hasattr(run, 'project') and hasattr(run, 'out'):
        return Path(run.project), Path(run.out)
    from google.colab import drive
    drive.mount('/content/drive')
    folder = Path('/content/drive/MyDrive/DSC261_UltraFeedback/full_500_recovery_v1')
    archives = [folder/name for name in ('frozen_inputs_and_pilot_eight_modes.zip', 'frozen_inputs_and_pilot.zip')]
    archive = next((p for p in archives if p.is_file()), None)
    if archive is None:
        raise FileNotFoundError(f'Frozen study inputs missing in {folder}. Run the full notebook setup with your experiment Drive, then rerun this cell.')
    project = Path(tempfile.mkdtemp(prefix='uf-analysis-inputs-'))
    # Extract only the frozen dependencies and data used by this analysis.
    names = ['src/'+n+'.py' for n in ('__init__','core','study','experiment','uf_trial')]
    names += ['config/judge_prompts.upstream.jsonl','full.jsonl','pilot.jsonl','configs.json','source_trial_manifest.json']
    with zipfile.ZipFile(archive) as z:
        for name in names:
            z.extract(name,project)
    return project, folder


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


def analyze(project, drive_folder, target, source_code=None):
    sys.path.insert(0,str(project))
    from src.core import read_jsonl, digest
    from src.study import semifinals_manifest, finals_manifest, h3_outcomes
    from src.uf_trial import _snapshot, baselines, validate_prompts
    from src.experiment import parse_verdict
    from matplotlib import pyplot as plt

    prompts=read_jsonl(project/'full.jsonl')
    pilot=read_jsonl(project/'pilot.jsonl')
    manifest=json.loads((project/'source_trial_manifest.json').read_text())
    configs=[c for c in json.loads((project/'configs.json').read_text()) if c['config_id'] in MODE_IDS]
    by_id={p['prompt_id']:p for p in prompts}
    pilot_ids={e['prompt_id'] for e in pilot}
    validate_prompts(prompts)
    if (len(prompts)!=500 or len(pilot_ids)!=25 or set(MODE_IDS)!={c['config_id'] for c in configs}
        or len(configs)!=6 or digest([p for p in prompts if p['prompt_id'] in pilot_ids])!=manifest['sample_hash']):
        raise ValueError('Sample or configurations differ from the frozen final study.')
    chunks={k.upper():v for k,v in CHUNKS.items()}
    contract=digest([prompts,configs,pilot,chunks,'full-v1'])
    if json.loads((drive_folder/'contract.json').read_text())!=contract:
        raise ValueError('Frozen inputs/settings do not match the Drive checkpoints.')

    target.mkdir(parents=True,exist_ok=True)
    coverage=[]; decisions=[]; attempts=[]; hardware=[]; inventory=[]
    drop={'content','reasoning','messages','environment'}
    for c in configs:
        cid=c['config_id']
        step=CHUNKS[cid.split('-')[1]]
        for start in range(0,500,step):
            part=start//step+1
            subset=prompts[start:start+step]
            ids={p['prompt_id'] for p in subset}
            folder=drive_folder/cid/f'part-{part:02}'
            events=[e for e in pilot if e['config_id']==cid and e['prompt_id'] in ids]
            for path in sorted(folder.glob('batch-*.json')):
                raw=path.read_bytes()
                if path.stem.split('-')[-1]!=hashlib.sha256(raw).hexdigest():
                    raise ValueError(f'Damaged checkpoint {path}')
                batch=json.loads(raw)
                inventory.append(dict(config_id=cid,part=part,file=str(path.relative_to(drive_folder)),
                                      bytes=len(raw),sha256=path.stem.split('-')[-1],attempts=len(batch)))
                events.extend(batch)
            unique={}; hashes={}
            for e in events:
                key=(e['request_id'],e['attempt'])
                fingerprint=digest(e)
                if key in unique and hashes[key]!=fingerprint:
                    raise ValueError(f'Conflicting copies of {key} in {folder}. Resolve before pooling.')
                unique[key]=e; hashes[key]=fingerprint
            ordered=sorted(unique.values(), key=lambda e:e['attempt'])
            latest={e['request_id']:e for e in ordered}
            semi=semifinals_manifest(subset,[c])
            finals,_=finals_manifest(subset,[c],[latest[r['request_id']] for r in semi if r['request_id'] in latest])
            expected={r['request_id']:r for r in semi+finals}
            for e in ordered:
                req=expected.get(e['request_id'])
                if req is None or any(e.get(k)!=v for k,v in req.items()):
                    raise ValueError(f'Unexpected request or switched candidates in {cid} part {part}.')
                if (e['status']=='ok' and e['verdict'] not in ('A','B')) or (e['status']!='ok' and e['verdict'] is not None):
                    raise ValueError('Inconsistent status and verdict.')
                env=e.get('environment',{})
                attempts.append(dict(config_id=cid,part=part,prompt_id=e['prompt_id'],request_id=e['request_id'],
                    attempt=e['attempt'],origin=e.get('origin'),status=e['status'],gpu=env.get('gpu'),revision=env.get('revision'),
                    allocated_seconds=e.get('allocated_seconds'),answer_tokens=e.get('answer_tokens'),reasoning_tokens=e.get('reasoning_tokens'),
                    forced_close=e.get('reasoning_forced_closed'),strict_label=parse_verdict(e.get('content'),e['order'],'terminal_ab')))
                if env: hardware.append((cid,json.dumps(env,sort_keys=True)))
            joined,status=_snapshot(subset,[c],ordered)
            st=status[0]
            coverage.append(dict(config_id=cid,part=part,**{k:v for k,v in st.items() if k not in ('config_id','last_attempt_statuses')}))
            for e in joined:
                decisions.append({k:v for k,v in e.items() if k not in drop})
        print('Checked',cid,flush=True)
    cov=pd.DataFrame(coverage)
    pending=cov[(cov.semifinals_pending+cov.finals_pending)>0]
    state='COMPLETE' if pending.empty else 'PARTIAL'
    cov.to_csv(target/'coverage_by_part.csv',index=False)
    grouped=cov.groupby('config_id').sum(numeric_only=True).drop(columns='part')
    grouped.to_csv(target/'coverage_by_mode.csv')
    pending.to_csv(target/'pending_parts.csv',index=False)
    att=pd.DataFrame(attempts)
    att.to_csv(target/'recorded_attempts.csv',index=False)
    frame_csv([dict(config_id=c,environment=json.loads(e),recorded_attempts=n) for (c,e),n in Counter(hardware).items()],target/'hardware.csv')
    compact=['config_id','prompt_id','request_id','candidate_a','candidate_b','match_name','order','status','verdict','attempt','origin']
    dec=pd.DataFrame(decisions)[compact]
    dec.to_csv(target/'latest_decisions.csv',index=False)
    stats=[]
    for cid in MODE_IDS:
        a=att[att.config_id==cid]; d=dec[dec.config_id==cid]
        last=d.merge(a[['request_id','attempt','strict_label']],on=['request_id','attempt'],validate='one_to_one')
        ok=last[last.status=='ok']
        stats.append(dict(config_id=cid,recorded_attempts=len(a),recorded_attempt1=sum(a.attempt==1),
            valid_latest=len(ok),invalid_latest=sum(d.status=='invalid_verdict'),truncated_latest=sum(d.status=='answer_truncated'),
            relaxed_valid_but_fails_strict_format=int((ok.strict_label!=ok.verdict).sum()),
            recorded_forced_closes=int(a.forced_close.fillna(False).sum()),recorded_generation_hours=a.allocated_seconds.sum()/3600))
    frame_csv(stats,target/'judge_diagnostics.csv')
    pd.DataFrame(inventory,columns=['config_id','part','file','bytes','sha256','attempts']).to_csv(target/'checkpoint_inventory.csv',index=False)
    # Both accepted presentations have verdicts in ORIGINAL identity space.
    matches=dec[dec.status=='ok'].pivot(index=['config_id','prompt_id','match_name'],columns='order',values='verdict').reindex(columns=['AB','BA']).dropna().reset_index()
    matches['order_disagreement']=matches.AB!=matches.BA
    matches['displayed_first_rate']=((matches.AB=='A').astype(float)+(matches.BA=='B').astype(float))/2
    matches.to_csv(target/'order_pairs.csv',index=False)
    match_rows=[]
    for cid in MODE_IDS:
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
    complete_ids={cid:set(n4[n4.config_id==cid].prompt_id) for cid in MODE_IDS}
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
        for cid in MODE_IDS:
            f=matched[(matched.config_id==cid)&matched.prompt_id.isin(ids)]
            for n in (1,2,4):
                sample=f[f.n==n]
                summaries+=estimates(sample,metrics,cohort=label,config_id=cid,n=n,eligible_prompts=len(ids))
        # Primary mode comparisons and exploratory adjacent-size contrasts use the same prompts on both sides.
        contrasts=[(MODE_IDS[i+1],MODE_IDS[i],'thinking_on_minus_off') for i in (0,2,4)]
        contrasts += [(MODE_IDS[i+2],MODE_IDS[i],'larger_minus_smaller') for i in range(4)]
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
        for i,cid in enumerate(MODE_IDS):
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
    stamp=datetime.now(timezone.utc).isoformat()
    metadata=dict(state=state,created_utc=stamp,study_contract=contract,scope=MODE_IDS,source_manifest=manifest,
        bootstrap_replicates=REPS,seed=SEED,configs=configs,common_all_six_prompts=len(common),
        planned_tournaments=3000,usable_tournaments=len(n4),pending_parts=len(pending),units={'format_drift':'fraction; multiply by 100 for percentage points'})
    (target/'manifest.json').write_text(json.dumps(metadata,indent=2))
    shutil.copy2(project/'full.jsonl',target/'frozen_inputs.jsonl')
    if source_code: (target/'analysis_cell.py').write_text(source_code)
    for name in ('source_trial_manifest.json','recovery_metadata.json','recovery_audit.jsonl'):
        if (project/name).is_file(): shutil.copy2(project/name,target/name)
    lines=[f'# UltraFeedback final-study analysis | {state}', '',
        f'500 frozen prompts, 4 answers each, 3 Qwen3 sizes, thinking off/on. {len(n4)}/3000 usable four-answer tournaments; {len(common)} prompts available in every mode. {len(pending)} parts have pending requests.',
        '', '## Best-of-4 Markdown drift','',
        '| Cohort | Observed prompts | Usable tournaments | Drift (pp) | 95% interval (pp) |',
        '|---|---:|---:|---:|---:|']
    def fmt(x):return 'NA' if pd.isna(x) else f'{100*x:+.2f}'
    for label,r in headline.iterrows():
        lines.append(f"| {label} | {int(r.prompts)} | {int(r.observations)} | {fmt(r['mean'])} | {fmt(r.ci_low)}, {fmt(r.ci_high)} |")
    h=headline.loc['all_500']
    conclusion=('Pending parts remain; use these values as a progress snapshot.' if state=='PARTIAL' else
        'The primary interval lies above zero, consistent with positive drift in the six observed modes.' if h.ci_low>0 else
        'The primary interval lies below zero, opposite to the proposed direction.' if h.ci_high<0 else
        'The primary interval includes zero or is unavailable; evidence for positive drift is inconclusive.')
    lines+=['',conclusion,'','## Results by mode (four candidates)','','| Mode | Usable / planned | Quality / 5 | Regret | Markdown drift (pp) | Best-rated selected (%) |','|---|---:|---:|---:|---:|---:|']
    def value(x,scale=1):
        return 'NA' if pd.isna(x) else f'{float(x)*scale:.3f}'
    main=summary[(summary.cohort=='all_500') & (summary.n==4)]
    for cid in MODE_IDS:
        mode=main[main.config_id==cid].set_index('metric')['mean']
        lines.append(f"| {cid} | {len(complete_ids[cid])}/500 | {value(mode.selected_quality)} | {value(mode.regret)} | {value(mode.format_drift,100)} | {value(mode.best_quality_selected,100)} |")
    lines+=['','## Paired thinking comparison (on minus off)','','| Model | Matched prompts | Quality difference | 95% interval |','|---|---:|---:|---:|']
    for r in pairs:
        if r['cohort']=='all_500' and r['contrast']=='thinking_on_minus_off' and r['metric']=='selected_quality':
            lines.append(f"| {r['left'].replace('-on','')} | {r['prompts']} | {value(r['mean'])} | {value(r['ci_low'])}, {value(r['ci_high'])} |")
    lines+=['','## Interpretation and handoff','',
        '- Primary estimate: selected Markdown indicator minus the proxy-best indicator at n=4, equally weighted over usable configuration-prompt tournaments. Bootstrap whole observed prompts with all their observed modes together (10,000 draws, percentile 95% intervals, seed 261). A one-prompt cohort has no interval.',
        '- The full 500 include the 25 pilot prompts once. new_475 excludes all 25 prompts used to refine the parser. common_six requires a complete outcome in every mode. no_gpt4_candidates excludes prompts with a GPT-4 candidate. Per-mode n=1/2/4 comparisons use the same complete n=4 prompts within each mode. Paired contrasts use the intersection for the two modes in question.',
        '- Quality is the mean of four GPT-4 fine-grained ratings (1–5). Regret is the best available rating minus the selected rating. A tied best rating counts as best selected. The seeded proxy-best answer determines Markdown drift in a tie. Zero n=1 regret follows from the single candidate.',
        '- An AB/BA disagreement is resolved using the frozen seeded answer-identity rule. Order diagnostics use both valid presentations of a match. Tournament disagreement differs from the Arena controlled bias test. No human votes or counterfactual format edits were collected here.',
        '- The 8B configuration was omitted from this study. Mode comparisons also change sampling parameters. The parsing rule was refined using the pilot. Missing outcomes may affect averages. Pairwise and source contrasts are exploratory, with unadjusted intervals. An Arena bias/drift correlation requires a separately matched Arena table.',
        '- Recorded attempt and time totals include latest saved pilot attempts plus full-run attempts. Earlier pilot retries, loading, idle time and session overhead are excluded. Strict-format diagnostics reparse observed answers only and do not simulate an independent strict-protocol run.',
        '- For merging, use prompt_outcomes.csv keyed by config_id, prompt_id and n within UltraFeedback. Missing outcome metrics are blank. Use pooled_h3.csv for primary and sensitivity estimates, per_mode.csv for curves, paired_comparisons.csv for matched differences, and pending_parts.csv for unfinished work. Raw responses remain in the source checkpoint folders and pilot bundle.']
    (target/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'+lines[0],f'| {len(n4)}/3000 usable tournaments | {len(pending)} pending parts')
    print(grouped[['valid_brackets','planned_brackets','semifinals_pending','finals_pending']].to_string())
    if len(pending):print('Pending parts:',pending.groupby('config_id').part.apply(list).to_dict())
    print('Pooled H3 (percentage points):',fmt(h['mean']), '[',fmt(h.ci_low),',',fmt(h.ci_high),']')
    return state


def export_six_modes():
    project,folder=locate()
    # Build locally; publish only a verified complete archive to mounted Drive.
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'UltraFeedback_six_modes'
        # IPython keeps the actual executed cell, including edits made by the user.
        history=globals().get('_ih',[])
        source=next((cell for cell in reversed(history) if 'def export_six_modes():' in cell),None)
        if not source and globals().get('__file__'):
            candidate=Path(__file__)
            if candidate.name=='ANALYZE_SIX_MODES.py' and candidate.is_file():
                source=candidate.read_text()
        state=analyze(project,folder,path,source)
        archive=Path(shutil.make_archive(str(path),'zip',root_dir=path.parent,base_dir=path.name))
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        dest=folder/f'UF_6_MODES_{state}_{stamp}_{uuid.uuid4().hex[:6]}.zip'
        tmp=dest.with_suffix('.uploading')
        shutil.copyfile(archive,tmp)
        if hashlib.sha256(tmp.read_bytes()).digest()!=hashlib.sha256(archive.read_bytes()).digest():
            raise IOError('Drive archive verification failed.')
        os.replace(tmp,dest)
    print('Share this verified Drive ZIP:',dest)
    return dest


if __name__=='__main__':
    analysis_zip = export_six_modes()