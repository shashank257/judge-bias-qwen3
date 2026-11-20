"""Small, standalone Colab UltraFeedback trial. No hosted API calls.

Reuses the project's pairwise prompt, bracket construction and identity mapping.
GPU inference is explicit; analysis never invents missing judgments.
"""
import csv
import itertools
import json
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .core import SEED, digest, read_jsonl, save_json, write_jsonl
from .experiment import parse_verdict
from .study import (semifinals_manifest, finals_manifest, h3_outcomes,
                    match_requests, stable_choice)

MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
MODEL_REVISIONS = {
    '0.6B': REVISION,
    '1.7B': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
    '4B': '1cfa9a7208912126459214e8b04321603b3df60c',
}


def trial_configs(precision='float16', batch_size=4, sizes=None):
    sizes = list(MODEL_REVISIONS) if sizes is None else list(sizes)
    if not sizes or len(set(sizes)) != len(sizes) or any(s not in MODEL_REVISIONS for s in sizes):
        raise ValueError('Choose unique sizes from 0.6B, 1.7B, 4B. 8B is excluded from this trial.')
    return [dict(config_id='qwen3-' + size.lower() + '-' + ('on' if thinking else 'off'),
                 checkpoint='Qwen/Qwen3-' + size, revision=MODEL_REVISIONS[size], thinking=thinking,
                 precision=precision, backend='transformers', batch_size=min(batch_size, 1) if size == '4B' else batch_size,
                 reasoning_budget=2048, answer_budget=512, max_context=32768,
                 temperature=0.6 if thinking else 0.7,
                 top_p=0.95 if thinking else 0.8, top_k=20, min_p=0.0,
                 invalid_retries=1, trial_version='uf-midterm-1')
            for size in sizes for thinking in [False, True]]


def select_trial(pool):
    """Preselect without looking at judge results; retain all four original answers."""
    selected = []
    for source, n in [('sharegpt', 12), ('evol_instruct', 7), ('ultrachat', 6)]:
        candidates = sorted((p for p in pool if p['source'] == source), key=lambda p: p['prompt_id'])
        selected.extend(random.Random(SEED).sample(candidates, n))
    random.Random(SEED).shuffle(selected)
    return selected


def validate_prompts(prompts):
    if len({p['prompt_id'] for p in prompts}) != len(prompts):
        raise ValueError('Duplicate prompt IDs')
    for p in prompts:
        ids = [a['answer_id'] for a in p['answers']]
        if len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError('Every prompt needs four distinct answer IDs')
        order = p['candidate_subsets']['4']
        if set(order) != set(ids) or p['candidate_subsets']['2'] != order[:2] or p['candidate_subsets']['1'] != order[:1]:
            raise ValueError('Candidate subsets must be nested')
        for a in p['answers']:
            if not a['text'].strip() or not 1 <= a['quality'] <= 5:
                raise ValueError('Invalid answer or quality score')
            if abs(a['quality'] - sum(a['ratings'].values()) / 4) > 1e-9:
                raise ValueError('Quality must be the mean of four GPT-4 ratings')


def _trim_at_stop(ids, stops):
    for i, token in enumerate(ids):
        if token in stops:
            return ids[:i + 1]
    return ids


class LocalJudge:
    def __init__(self, configs):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('Select Runtime > Change runtime type > GPU, then reconnect.')
        model, revision = configs[0]['checkpoint'], configs[0]['revision']
        if any((c['checkpoint'], c['revision']) != (model, revision) for c in configs):
            raise ValueError('Load only one model at a time; use run_models for multiple sizes.')
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        precision = configs[0]['precision']
        if any(c['precision'] != precision for c in configs):
            raise ValueError('Both modes must use the same precision')
        dtype = torch.float16 if precision == 'float16' else torch.bfloat16
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError('This GPU does not support BF16; use float16 for both modes.')
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation='sdpa').to('cuda').eval()
        self.think_end = self.tokenizer.convert_tokens_to_ids('</think>')
        self.eos = self.model.generation_config.eos_token_id
        self.eos = [self.eos] if isinstance(self.eos, int) else list(self.eos)
        self.metadata = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                         'transformers': transformers.__version__, 'model': model,
                         'revision': revision, 'precision': precision}

    def close(self):
        import gc
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()

    def encode(self, row, config):
        return self.tokenizer.apply_chat_template(row['messages'], tokenize=True,
                    add_generation_prompt=True, enable_thinking=config['thinking'])

    def preflight(self, prompts, configs):
        longest = 0
        for c in configs:
            for p in prompts:
                for a, b in itertools.combinations(p['candidate_subsets']['4'], 2):
                    for row in match_requests(p, c, a, b, 'preflight'):
                        length = len(self.encode(row, c))
                        longest = max(longest, length)
                        reserve = c['answer_budget'] + (c['reasoning_budget'] + 8 if c['thinking'] else 0)
                        if length + reserve > c['max_context']:
                            raise ValueError('A selected prompt exceeds the context limit. Do not silently truncate it.')
        return {'max_prompt_tokens': longest, 'context_limit': configs[0]['max_context']}

    def _generate(self, sequences, config, budget, stops):
        torch = self.torch
        batch = self.tokenizer.pad({'input_ids': sequences}, padding=True, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            output = self.model.generate(**batch, do_sample=True, num_beams=1,
                temperature=config['temperature'], top_p=config['top_p'], top_k=config['top_k'],
                min_p=0.0, repetition_penalty=1.0, max_new_tokens=budget,
                eos_token_id=stops, pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True)
        generated = output[:, batch['input_ids'].shape[1]:].cpu().tolist()
        return [_trim_at_stop(ids, set(stops)) for ids in generated]

    def generate_batch(self, rows, config, attempt):
        torch = self.torch
        batch_seed = int(digest([SEED, [r['request_id'] for r in rows], attempt])[:8], 16)
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        inputs = [self.encode(r, config) for r in rows]
        reasons = [[] for _ in rows]
        forced = [False] * len(rows)
        continued = inputs
        torch.cuda.synchronize()
        began = time.perf_counter()
        if config['thinking']:
            reasons = self._generate(inputs, config, config['reasoning_budget'], [self.think_end] + self.eos)
            continued = []
            for i, (original, reason) in enumerate(zip(inputs, reasons)):
                if reason and reason[-1] in self.eos:
                    reason = reason[:-1]
                    reasons[i] = reason
                forced[i] = not reason or reason[-1] != self.think_end
                # Close at 2,048 generated reasoning tokens; never fabricate a verdict.
                close = [self.think_end] if forced[i] else []
                continued.append(original + reason + close + self.tokenizer.encode('\n\n', add_special_tokens=False))
        answers = self._generate(continued, config, config['answer_budget'], self.eos)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        results = []
        for i, ids in enumerate(answers):
            content = self.tokenizer.decode(ids, skip_special_tokens=True)
            ended = bool(ids) and ids[-1] in self.eos
            if config.get('parser_policy') == 'consistent_ab_anywhere_v1':
                from .uf_recovery import recover_verdict
                verdict = recover_verdict(content, rows[i]['order'])
            else:
                verdict = parse_verdict(content, rows[i]['order'], 'terminal_ab')
            status = 'ok' if verdict and ended else 'answer_truncated' if not ended else 'invalid_verdict'
            results.append(dict(status=status, verdict=verdict if status == 'ok' else None,
                content=content, reasoning=self.tokenizer.decode(reasons[i], skip_special_tokens=False),
                reasoning_forced_closed=forced[i], reasoning_tokens=len(reasons[i]),
                prompt_tokens=len(inputs[i]), answer_tokens=len(ids),
                batch_seed=batch_seed, batch_seconds=seconds, batch_size=len(rows),
                allocated_seconds=seconds / len(rows), attempt=attempt))
        return results


def _terminal(event):
    return bool(event) and not (event['status'] == 'invalid_verdict' and event['attempt'] == 0)


def _snapshot(prompts, configs, events):
    latest = {e['request_id']: e for e in events}
    joined, status = [], []
    for c in configs:
        semis = semifinals_manifest(prompts, [c])
        sr = [latest[r['request_id']] for r in semis if r['request_id'] in latest]
        finals, missing = finals_manifest(prompts, [c], sr)
        fr = [latest[r['request_id']] for r in finals if r['request_id'] in latest]
        pending_s = sum(not _terminal(latest.get(r['request_id'])) for r in semis)
        pending_f = sum(not _terminal(latest.get(r['request_id'])) for r in finals)
        outcomes = h3_outcomes(prompts, [c], sr + fr)
        valid_brackets = sum(r['n'] == 4 and not r['missing'] for r in outcomes)
        state = 'NOT STARTED' if not sr else 'PENDING' if pending_s or pending_f else 'FINISHED'
        if state == 'FINISHED' and valid_brackets < len(prompts):
            state = 'FINISHED WITH MISSING BRACKETS'
        status.append({'config_id': c['config_id'], 'state': state,
                       'semifinals_saved': len(sr), 'semifinals_valid': sum(r['status'] == 'ok' for r in sr),
                       'semifinals_pending': pending_s, 'finals_eligible': len(finals),
                       'finals_saved': len(fr), 'finals_valid': sum(r['status'] == 'ok' for r in fr),
                       'finals_pending': pending_f, 'valid_brackets': valid_brackets,
                       'planned_brackets': len(prompts),
                       'last_attempt_statuses': dict(Counter(r['status'] for r in sr + fr))})
        joined.extend(sr + fr)
    return joined, status


def collect_progress(prompts, configs, output_dir, export=False):
    """Read authoritative per-model logs, including an interrupted legacy run."""
    out = Path(output_dir)
    suite_meta = out / 'suite_metadata.json'
    signature = digest({'prompts': prompts, 'configs': configs})
    if suite_meta.exists() and json.loads(suite_meta.read_text())['signature'] != signature:
        raise ValueError('Settings do not match this saved run. Restore N_PROMPTS/MODEL_SIZES/BATCH_SIZE/RUN_NAME.')
    all_rows, all_events, statuses = [], [], []
    for model in dict.fromkeys(c['checkpoint'] for c in configs):
        group = [c for c in configs if c['checkpoint'] == model]
        target = out / model.split('/')[-1]
        path = target / 'judge_results.jsonl'
        events = read_jsonl(path) if path.exists() else []
        expected = digest({'prompts': prompts, 'configs': group})
        if any(e['run_signature'] != expected for e in events):
            raise ValueError('Saved model settings differ; do not mix results.')
        rows, report = _snapshot(prompts, group, events)
        all_rows.extend(rows)
        all_events.extend(events)
        statuses.extend(report)
    if export:
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / 'joined_results.jsonl', all_rows)
        write_jsonl(out / 'judge_results.jsonl', all_events)
        write_jsonl(out / 'selected_prompts.jsonl', prompts)
        save_json(out / 'progress.json', statuses)
    return all_rows, statuses


def diagnostic_report(prompts, configs, output_dir, limit=3):
    """Preserve raw outputs for diagnosis; never change the verdict parser here."""
    import re
    out = Path(output_dir)
    rows, progress = collect_progress(prompts, configs, out, export=True)
    report = {'progress': progress, 'examples': [],
              'note': 'No reclassification or re-inference. Inspect output text before changing the protocol.'}
    for c in configs:
        failed = [r for r in rows if r['config_id'] == c['config_id'] and r['status'] != 'ok'][:limit]
        for r in failed:
            content = r.get('content', '')
            report['examples'].append({
                'config_id': c['config_id'], 'request_id': r['request_id'], 'status': r['status'],
                'attempt': r['attempt'], 'content': content,
                'bracketed_labels': re.findall(r'\[\[([^\]]+)\]\]', content),
                'reasoning_tail': r.get('reasoning', '')[-1200:],
                'prompt_tokens': r.get('prompt_tokens'), 'answer_tokens': r.get('answer_tokens'),
                'reasoning_forced_closed': r.get('reasoning_forced_closed'), 'environment': r.get('environment')})
    save_json(out / 'diagnostics.json', report)
    return report


def run_trial(prompts, configs, judge, output_dir, progress=print,
              only_config_ids=None, max_new_calls=None, max_seconds=None):
    """Flush each completed batch; rerunning resumes from completed requests."""
    validate_prompts(prompts)
    if len({c['checkpoint'] for c in configs}) != 1:
        raise ValueError('run_trial accepts one loaded model; use run_models for several models.')
    selected = {c['config_id'] for c in configs} if only_config_ids is None else set(only_config_ids)
    if not selected.issubset({c['config_id'] for c in configs}):
        raise ValueError('Unknown requested mode')
    if max_new_calls is not None and max_new_calls <= 0:
        raise ValueError('max_new_calls must be positive or None')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    signature = digest({'prompts': prompts, 'configs': configs})
    meta_path = out / 'run_metadata.json'
    previous = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if previous and previous['signature'] != signature:
        raise ValueError('Run settings/data changed. Use a new RUN_NAME to keep results separate.')
    environments = previous.get('environments', [])
    if judge.metadata not in environments:
        environments.append(judge.metadata)
    save_json(meta_path, {'signature': signature, 'configs': configs, 'environment': judge.metadata,
                         'environments': environments,
                         'sample_size': len(prompts), 'sample_hash': digest(prompts),
                         'note': 'Small independent midterm trial; no claim of matching teammate frozen inputs.'})
    write_jsonl(out / 'selected_prompts.jsonl', prompts)
    log_path = out / 'judge_results.jsonl'
    events = read_jsonl(log_path) if log_path.exists() else []
    if any(e['run_signature'] != signature for e in events):
        raise ValueError('Result log belongs to another run')
    latest = {e['request_id']: e for e in events}
    new_calls = 0
    began = time.monotonic()

    def paused():
        return ((max_new_calls is not None and new_calls >= max_new_calls) or
                (max_seconds is not None and time.monotonic() - began >= max_seconds))

    def execute(rows, config, stage):
        nonlocal new_calls
        for attempt in [0, 1]:
            todo = [r for r in rows if r['request_id'] not in latest] if attempt == 0 else [
                r for r in rows if latest.get(r['request_id'], {}).get('status') == 'invalid_verdict'
                and latest[r['request_id']]['attempt'] == 0]
            start = 0
            while start < len(todo) and not paused():
                count = config['batch_size']
                if max_new_calls is not None:
                    count = min(count, max_new_calls - new_calls)
                batch = todo[start:start + count]
                generated = judge.generate_batch(batch, config, attempt)
                if len(generated) != len(batch):
                    raise ValueError('Generator returned the wrong batch length')
                with log_path.open('a') as f:
                    for row, result in zip(batch, generated):
                        event = dict(row, **result, run_signature=signature, environment=judge.metadata,
                                     utc=datetime.now(timezone.utc).isoformat())
                        f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
                        latest[row['request_id']] = event
                    f.flush()
                    os.fsync(f.fileno())
                start += len(batch)
                new_calls += len(batch)
                current = [latest[r['request_id']] for r in rows if r['request_id'] in latest]
                valid = sum(r['status'] == 'ok' for r in current)
                pending = sum(not _terminal(latest.get(r['request_id'])) for r in rows)
                progress(f"{config['config_id']} | {stage} | {'first pass' if attempt == 0 else 'retry'} "
                         f"{start}/{len(todo)} | valid {valid}/{len(rows)} | pending {pending} | new calls this cell {new_calls}")
        return [latest[r['request_id']] for r in rows if r['request_id'] in latest]

    all_rows = []
    for config in configs:
        if config['config_id'] not in selected:
            continue
        semis = semifinals_manifest(prompts, [config])
        semi_results = execute(semis, config, 'semifinals')
        finals, missing = finals_manifest(prompts, [config], semi_results)
        final_results = execute(finals, config, 'finals')
        all_rows.extend(semi_results + final_results)
        if missing and not paused():
            progress(f"{len(missing)} brackets lack a usable semifinal: no winner will be invented.")
    # Rebuild from ALL saved modes, so running one cell never discards the other mode.
    all_rows, status = _snapshot(prompts, configs, list(latest.values()))
    write_jsonl(out / 'joined_results.jsonl', all_rows)
    save_json(out / 'progress.json', status)
    if paused():
        progress('Chunk saved. Re-run this same mode cell to continue; completed judgments will be skipped.')
    return all_rows


def run_one_mode(prompts, configs, output_dir, config_id, judge_factory=LocalJudge,
                 max_new_calls=40, max_seconds=600, progress=print):
    """Compatible with old two-mode hashes/folders; execute only one requested mode."""
    rows, statuses = collect_progress(prompts, configs, output_dir)
    current = next(s for s in statuses if s['config_id'] == config_id)
    if current['state'].startswith('FINISHED'):
        progress(f"{config_id}: {current['state']}; {current['valid_brackets']}/{len(prompts)} usable brackets. No generation needed.")
        return rows, statuses
    c = next(c for c in configs if c['config_id'] == config_id)
    group = [other for other in configs if other['checkpoint'] == c['checkpoint']]
    target = Path(output_dir) / c['checkpoint'].split('/')[-1]
    judge = judge_factory(group)
    try:
        progress(str(judge.metadata))
        progress('Context checks: ' + str(judge.preflight(prompts, [c])))
        run_trial(prompts, group, judge, target, progress, only_config_ids=[config_id],
                  max_new_calls=max_new_calls, max_seconds=max_seconds)
    finally:
        judge.close()
    return collect_progress(prompts, configs, output_dir, export=True)


def run_models(prompts, configs, output_dir, judge_factory=LocalJudge, progress=print):
    """Finish both modes for one model, release its memory, then load the next."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    signature = digest({'prompts': prompts, 'configs': configs})
    meta = out / 'suite_metadata.json'
    if meta.exists() and json.loads(meta.read_text())['signature'] != signature:
        raise ValueError('Suite settings changed. Choose a new run name.')
    save_json(meta, {'signature': signature, 'configs': configs, 'sample_hash': digest(prompts)})
    all_rows, attempts = [], []
    for checkpoint in dict.fromkeys(c['checkpoint'] for c in configs):
        group = [c for c in configs if c['checkpoint'] == checkpoint]
        target = out / checkpoint.split('/')[-1]
        group_signature = digest({'prompts': prompts, 'configs': group})
        complete = target / 'completed.json'
        if complete.exists():
            saved = json.loads(complete.read_text())
            rows = read_jsonl(target / 'joined_results.jsonl')
            if saved != {'signature': group_signature, 'results_hash': digest(rows)}:
                raise ValueError('Completed model files or settings changed; use a new run name.')
            progress(checkpoint + ': already finished; skipping model download/load.')
        else:
            progress('Loading ' + checkpoint + ' (both thinking modes; one model in GPU memory).')
            judge = judge_factory(group)
            try:
                progress(str(judge.metadata))
                progress('Context checks: ' + str(judge.preflight(prompts, group)))
                rows = run_trial(prompts, group, judge, target, progress)
                save_json(complete, {'signature': group_signature, 'results_hash': digest(rows)})
            finally:
                judge.close()
                del judge
        all_rows.extend(rows)
        attempts.extend(read_jsonl(target / 'judge_results.jsonl'))
        # Keep a usable combined checkpoint even if the next model is interrupted.
        write_jsonl(out / 'joined_results.jsonl', all_rows)
        write_jsonl(out / 'judge_results.jsonl', attempts)
    write_jsonl(out / 'selected_prompts.jsonl', prompts)
    return all_rows


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_ci(values):
    import numpy as np
    if not values:
        return None, None, None
    a = np.array(values, dtype=float)
    draws = np.random.default_rng(SEED).choice(a, size=(2000, len(a)), replace=True).mean(axis=1)
    return float(a.mean()), *map(float, np.quantile(draws, [.025, .975]))


def baselines(prompts):
    rows = []
    for p in prompts:
        by_id = {a['answer_id']: a for a in p['answers']}
        for n in [1, 2, 4]:
            answers = [by_id[i] for i in p['candidate_subsets'][str(n)]]
            best = max(a['quality'] for a in answers)
            avg = sum(a['quality'] for a in answers) / n
            rows.append({'prompt_id': p['prompt_id'], 'n': n, 'oracle_quality': best,
                         'random_expected_quality': avg, 'random_expected_regret': best - avg})
    return rows


def analyze_trial(prompts, configs, rows, output_dir, make_plots=True):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outcomes = h3_outcomes(prompts, configs, rows)
    for row in outcomes:
        row['best_quality_selected'] = None if row['missing'] else int(abs(row['regret']) < 1e-9)
    _csv(out / 'outcomes.csv', outcomes)
    base = baselines(prompts)
    _csv(out / 'baselines.csv', base)
    summary = []
    for c in configs:
        # Compare n=1,2,4 on the same complete prompt cohort within each mode.
        complete = {r['prompt_id'] for r in outcomes if r['config_id'] == c['config_id'] and r['n'] == 4 and not r['missing']}
        for sensitivity in [False, True]:
            for n in [1, 2, 4]:
                group = [r for r in outcomes if r['config_id'] == c['config_id'] and r['n'] == n
                         and r['prompt_id'] in complete and (not sensitivity or r['without_gpt4'])]
                item = {'config_id': c['config_id'], 'model': c['checkpoint'], 'thinking': c['thinking'], 'n': n,
                        'subset': 'without_gpt4_candidates' if sensitivity else 'all',
                        'planned_prompts': sum(not sensitivity or p['without_gpt4'] for p in prompts),
                        'complete_prompts': len(group)}
                for metric in ['selected_quality', 'regret', 'format_drift', 'best_quality_selected']:
                    mean, low, high = _mean_ci([r[metric] for r in group])
                    item.update({metric: mean, metric + '_ci_low': low, metric + '_ci_high': high})
                ids = {r['prompt_id'] for r in group}
                b = [r for r in base if r['n'] == n and r['prompt_id'] in ids]
                item['random_expected_quality'] = sum(r['random_expected_quality'] for r in b) / len(b) if b else None
                item['oracle_quality'] = sum(r['oracle_quality'] for r in b) / len(b) if b else None
                summary.append(item)
    _csv(out / 'summary.csv', summary)
    attempts = read_jsonl(out / 'judge_results.jsonl') if (out / 'judge_results.jsonl').exists() else rows
    run_stats = []
    for c in configs:
        final = [r for r in rows if r['config_id'] == c['config_id']]
        calls = [r for r in attempts if r['config_id'] == c['config_id']]
        run_stats.append({'config_id': c['config_id'], 'planned_max_judgments': 6 * len(prompts),
                          'attempts': len(calls), 'usable_judgments': sum(r['status'] == 'ok' for r in final),
                          'invalid_judgments': sum(r['status'] == 'invalid_verdict' for r in final),
                          'answer_truncated': sum(r['status'] == 'answer_truncated' for r in final),
                          'reasoning_forced_closed_attempts': sum(r.get('reasoning_forced_closed', False) for r in calls),
                          'generation_seconds': sum(r.get('allocated_seconds', 0) for r in calls)})
    _csv(out / 'run_stats.csv', run_stats)
    # A comparison of thinking modes must use the same prompts, not two different missingness cohorts.
    n4 = {c['config_id']: {r['prompt_id']: r for r in outcomes if r['config_id'] == c['config_id'] and r['n'] == 4 and not r['missing']} for c in configs}
    common = set.intersection(*(set(v) for v in n4.values())) if n4 else set()
    paired = {'common_prompts': len(common), 'models': {},
              'note': 'Descriptive small-sample comparison; sampling settings also differ by thinking mode.'}
    for checkpoint in dict.fromkeys(c['checkpoint'] for c in configs):
        modes = {c['thinking']: c['config_id'] for c in configs if c['checkpoint'] == checkpoint}
        if set(modes) != {False, True}:
            continue
        shared = sorted(set(n4[modes[False]]) & set(n4[modes[True]]))
        entry = {'common_prompts': len(shared)}
        for metric in ['selected_quality', 'regret', 'format_drift']:
            entry[metric + '_on_minus_off'] = _mean_ci([n4[modes[True]][p][metric] - n4[modes[False]][p][metric] for p in shared])
        paired['models'][checkpoint] = entry
    save_json(out / 'paired_thinking_comparison.json', paired)
    main = [r for r in summary if r['subset'] == 'all']
    lines = ['# UltraFeedback midterm trial', '',
             'Preliminary results from actual recorded judgments. GPT-4 ratings are a quality proxy, not human ground truth.', '',
             f"Sample: {len(prompts)} prompts with four existing answers each; models: {', '.join(dict.fromkeys(c['checkpoint'] for c in configs))}; thinking off/on.",
             'Data is independently sampled from this workspace, not verified against the teammate frozen dataset.',
             'No new answers are generated: the model only judges the provided answers.', '',
             '| Model | Mode | Candidates | Complete prompts | Selected quality (1-5) | Regret (lower is better) | Format drift (points) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in main:
        fmt = lambda value: 'NA' if value is None else f'{value:.3f}'
        drift = None if r['format_drift'] is None else 100 * r['format_drift']
        lines.append(f"| {r['model']} | {'on' if r['thinking'] else 'off'} | {r['n']} | {r['complete_prompts']} | {fmt(r['selected_quality'])} | {fmt(r['regret'])} | {fmt(drift)} |")
    lines += ['', 'Interpretation: regret is the best available GPT-4 proxy score minus the selected score.',
              'Format drift is the Markdown indicator of the chosen answer minus that of the proxy-best answer. It is descriptive, not proof that formatting caused the selection.',
              'n=1 is the seeded first answer; n=2 uses semifinal 1; n=4 uses the tournament final. No extra model calls for n=1/2.',
              'Both orders are judged; disagreement is resolved by a fixed seeded answer-identity priority. Missing/invalid matches do not produce invented winners.',
              'The same complete n=4 cohort is used for all n within each mode. Compare modes using paired_thinking_comparison.json.',
              '95% bootstrap intervals (2,000 resamples, seed 261) are in summary.csv. At 25 prompts they are exploratory.',
              'This trial cannot establish model-size effects, replicate Arena bias estimates, or prove a causal link between formatting and quality.',
              'Transformers/Colab and recorded numerical precision differ from the teammate MLX run. Exact teammate inputs and prompt revision were unavailable.',
              'Thinking uses temperature 0.6/top-p 0.95; off uses 0.7/0.8. A separate sampling control has not been run.',
              'AI assistance: Codex prepared the trial code, data packaging and report template; results require actual GPU execution.', '',
              'Sources: https://huggingface.co/datasets/openbmb/UltraFeedback ; ' + ' ; '.join('https://huggingface.co/' + m for m in dict.fromkeys(c['checkpoint'] for c in configs))]
    (out / 'MIDTERM_SUMMARY.md').write_text('\n'.join(lines) + '\n')
    if make_plots and any(r['complete_prompts'] for r in main):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        models = list(dict.fromkeys(c['checkpoint'] for c in configs))
        fig, grid = plt.subplots(len(models), 3, figsize=(13, 3.5 * len(models)), squeeze=False)
        for c in configs:
            axes = grid[models.index(c['checkpoint'])]
            color = '#c2410c' if c['thinking'] else '#2563eb'
            group = [r for r in main if r['config_id'] == c['config_id'] and r['complete_prompts']]
            for ax, metric, title, scale in zip(axes, ['selected_quality', 'regret', 'format_drift'],
                    ['Selected proxy quality (1-5)', 'Selection regret (lower is better)', 'Format drift (percentage points)'], [1, 1, 100]):
                x = [r['n'] for r in group]
                y = [r[metric] * scale for r in group]
                low = [max(0, (r[metric] - r[metric + '_ci_low']) * scale) for r in group]
                high = [max(0, (r[metric + '_ci_high'] - r[metric]) * scale) for r in group]
                ax.errorbar(x, y, yerr=[low, high], marker='o', capsize=3, label='Thinking ' + ('on' if c['thinking'] else 'off'), color=color)
                ax.set(title=title, xlabel='Number of candidate answers', xticks=[1, 2, 4])
                ax.grid(alpha=.2)
        for model, axes in zip(models, grid):
            axes[0].set_ylim(.8, 5.2)
            axes[0].set_ylabel(model.split('/')[-1])
            axes[2].axhline(0, color='gray', linewidth=1)
            axes[0].legend(fontsize=8)
        fig.suptitle(f'UltraFeedback pilot: {len(prompts)} prompts, {len(models)} model sizes (exploratory 95% intervals)', fontsize=11)
        fig.tight_layout()
        fig.savefig(out / 'ultrafeedback_midterm.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    return summary
