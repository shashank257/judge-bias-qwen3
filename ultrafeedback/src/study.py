"""Proposal-aligned experimental design v2. No API calls in this module.

Ten format requests/pair/config = eight presentations + two MM noise repeats.
Position requests are reused wherever the independently selected pools overlap.
"""
import itertools
import json
from collections import defaultdict
from pathlib import Path
from functools import lru_cache
from .core import SEED, digest, sample

PARAPHRASE = '''Evaluate which of two assistants gives the better answer to the user. Compare their
accuracy, relevance, helpfulness, depth, creativity, and detail. Briefly explain your comparison.
Ignore which assistant appears first, the assistant names, and answer length by itself.
Choose a winner even if the answers are close. Finish with exactly [[A]] for assistant A
or [[B]] for assistant B. Do not output any other bracketed verdict.'''


@lru_cache(maxsize=1)
def _templates():
    return [json.loads(line) for line in (Path(__file__).resolve().parents[1] / 'config/judge_prompts.upstream.jsonl').read_text().splitlines()]


def judge_messages(record, condition='MM', order='AB', prompt_variant='primary'):
    upstream = _templates()
    multiturn = record.get('turn') == 2
    name = 'pair-v2-multi-turn' if multiturn else 'pair-v2'
    template = next(p for p in upstream if p['name'] == name)
    system = template['system_prompt'].replace(', and "[[C]]" for a tie', '')
    if prompt_variant == 'paraphrase':
        system = PARAPHRASE
    elif prompt_variant != 'primary':
        raise ValueError('Unknown prompt variant')
    values = {}
    if multiturn:
        if condition != 'MM':
            raise ValueError('MT-Bench histories are never format manipulated')
        values.update(question_1=record['conversation_a'][0]['content'], question_2=record['conversation_a'][2]['content'])
        for displayed, original in zip('ab', order.lower()):
            history = record['conversation_' + original]
            for turn in [1, 2]:
                values[f'answer_{displayed}_{turn}'] = history[2 * turn - 1]['content']
    else:
        answers = {identity: record[('answer_' if mode == 'M' else 'plain_') + identity.lower()]
                   for identity, mode in zip('AB', condition)}
        values = {'question': record['question'], 'answer_a': answers[order[0]], 'answer_b': answers[order[1]]}
    # User data is inserted through format arguments, never evaluated as instructions/code.
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': template['prompt_template'].format(**values)}]


def request(record, config, condition='MM', order='AB', repeat=0, prompt_variant='primary', experiment='position'):
    messages = judge_messages(record, condition, order, prompt_variant)
    return {'request_id': digest([record['pair_id'], messages, config, repeat]),
            'input_hash': digest(messages), 'pair_id': record['pair_id'], 'prompt_id': record['prompt_id'],
            'source': record['source'], 'condition': condition, 'order': order, 'repeat': repeat,
            'prompt_variant': prompt_variant, 'config_id': config['config_id'], 'config_hash': digest(config),
            'human_preference': record.get('human_preference'), 'messages': messages, 'experiments': [experiment]}


def bias_manifest(pools, configs, prompt_variant='primary'):
    unique = {}
    for experiment, records in pools.items():
        if experiment not in ['position', 'format']:
            raise ValueError('Unknown bias experiment')
        for record in records:
            conditions = ['MM'] if experiment == 'position' else ['MM', 'MP', 'PM', 'PP']
            for condition, order, config in itertools.product(conditions, ['AB', 'BA'], configs):
                for repeat in ([0, 1] if condition == 'MM' else [0]):
                    row = request(record, config, condition, order, repeat, prompt_variant, experiment)
                    key = row['request_id']
                    if key in unique:
                        unique[key]['experiments'].append(experiment)
                    else:
                        unique[key] = row
    import random
    rows = list(unique.values())
    random.Random(SEED).shuffle(rows)
    return rows


def mtbench_manifest(records, configs):
    pairs = {r['pair_id']: r for r in records}
    return [request(r, c, order=o, experiment='mtbench')
            for r, c, o in itertools.product(pairs.values(), configs, ['AB', 'BA'])]


def control_manifests(format_pool, configs):
    # The same 200-pair subset provides matched comparisons to the main experiment.
    subset = sample(format_pool, 200)
    pools = {'position': subset, 'format': subset}
    controls = []
    for c in configs:
        if c['thinking']:
            continue
        c = json.loads(json.dumps(c))
        c['config_id'] += '-sampling-control'
        c['sampling'].update(temperature=0.6, top_p=0.95, top_k=20, min_p=0)
        c['sampling_control'] = True
        # Controls must themselves have endpoint parameter support verified.
        c['controls_verified'] = False
        controls.append(c)
    para_configs = [c for c in configs if c['checkpoint'] in ['Qwen/Qwen3-1.7B', 'Qwen/Qwen3-8B']]
    return {'sampling': bias_manifest(pools, controls),
            'paraphrase': bias_manifest(pools, para_configs, 'paraphrase')}, controls


def join_results(manifest, events):
    latest = {}
    for event in events:
        if event['event'] == 'result':
            latest[event['request_id']] = event
    rows = []
    for row in manifest:
        event = latest.get(row['request_id'], {})
        result = dict(row)
        result.update({k: v for k, v in event.items() if k not in row})
        if not event or event.get('retry_pending', event.get('retry_scheduled', False)):
            result.update(status='pending', verdict=None)
        rows.append(result)
    return rows


def _flip(a, b, exclude_invalid):
    states = [x.get('status') if x else 'pending' for x in [a, b]]
    if any(s not in ['ok', 'invalid_verdict'] for s in states):
        return None  # HTTP/transport/truncation are not silently called parse failures.
    if 'invalid_verdict' in states:
        return None if exclude_invalid else 1.0
    return float(a['verdict'] != b['verdict'])


def bias_comparisons(rows, exclude_invalid=False):
    """Binary comparison rows with equal total weight per pair/bias/condition.

    Format edges strip only one answer at a time. Repeat MM is the common noise
    reference; that interpretation must be frozen after the pilot.
    """
    groups = defaultdict(dict)
    for r in rows:
        groups[(r['config_id'], r['pair_id'], r.get('prompt_variant', 'primary'))][r['condition'], r['order'], r['repeat']] = r
    comparisons, summaries = [], []
    edges = [('MM', 'MP'), ('MM', 'PM'), ('MP', 'PP'), ('PM', 'PP')]
    for (cfg, pid, variant), g in groups.items():
        first = next(iter(g.values()))
        experiments = {e for r in g.values() for e in r['experiments']}
        get = lambda c, o, k=0: g.get((c, o, k))
        noise = [_flip(get('MM', o, 0), get('MM', o, 1), exclude_invalid) for o in ['AB', 'BA']]
        for bias in ['position', 'format']:
            if bias not in experiments:
                continue
            edit = ([_flip(get('MM', 'AB', k), get('MM', 'BA', k), exclude_invalid) for k in [0, 1]]
                    if bias == 'position' else
                    [_flip(get(a, o), get(b, o), exclude_invalid) for a, b in edges for o in ['AB', 'BA']])
            # Keep edit and noise on the same complete pair population in both analyses.
            if any(v is None for v in edit + noise):
                continue
            base = {'pair_id': pid, 'prompt_id': first['prompt_id'], 'config_id': cfg,
                    'source': first['source'], 'bias': bias, 'prompt_variant': variant}
            edit_mean, noise_mean = sum(edit) / len(edit), sum(noise) / len(noise)
            for edited, values in [(1, edit), (0, noise)]:
                comparisons.extend(dict(base, edited=edited, flip=v, comparison_index=i,
                                        weight=1 / len(values)) for i, v in enumerate(values))
            signed = []
            cells = [('MM', o, 0) for o in ['AB', 'BA']] if bias == 'position' else [(c, o, 0) for c in ['MP', 'PM'] for o in ['AB', 'BA']]
            for c, o, k in cells:
                r = get(c, o, k)
                if r and r.get('status') == 'ok':
                    favored = o[0] if bias == 'position' else 'A' if c == 'MP' else 'B'
                    signed.append(float(r['verdict'] == favored))
            summaries.append(dict(base, edit_flip=edit_mean, noise_flip=noise_mean,
                                  excess_flip=edit_mean - noise_mean,
                                  signed_preference=sum(signed) / len(signed) if signed else None))
    return comparisons, summaries


def stable_choice(ids, key):
    """A fixed seeded random priority, independent of answer format/quality/position."""
    return min(ids, key=lambda i: digest([SEED, key, i]))


def match_record(prompt, candidate_a, candidate_b, match_name):
    answers = {a['answer_id']: a for a in prompt['answers']}
    return {'pair_id': digest(['h3', prompt['prompt_id'], match_name, candidate_a, candidate_b]),
            'prompt_id': prompt['prompt_id'], 'source': prompt['source'], 'question': prompt['question'],
            'answer_a': answers[candidate_a]['text'], 'answer_b': answers[candidate_b]['text'],
            'candidate_a': candidate_a, 'candidate_b': candidate_b, 'match_name': match_name,
            'human_preference': None}


def match_requests(prompt, config, candidate_a, candidate_b, match_name):
    rec = match_record(prompt, candidate_a, candidate_b, match_name)
    return [dict(request(rec, config, order=o, experiment='h3'), candidate_a=candidate_a,
                 candidate_b=candidate_b, match_name=match_name) for o in ['AB', 'BA']]


def semifinals_manifest(prompts, configs):
    rows = []
    for p, c in itertools.product(prompts, configs):
        ids = p['candidate_subsets']['4']
        for name, pair in [('semi0', ids[:2]), ('semi1', ids[2:])]:
            rows.extend(match_requests(p, c, *pair, name))
    return rows


def bracket_winners(rows):
    groups = defaultdict(dict)
    for r in rows:
        groups[(r['prompt_id'], r['config_id'], r['match_name'])][r['order']] = r
    winners = {}
    for key, g in groups.items():
        if set(g) != {'AB', 'BA'} or any(r.get('status') != 'ok' for r in g.values()):
            continue
        r = g['AB']
        ids = [r['candidate_a'], r['candidate_b']]
        votes = [ids['AB'.index(x['verdict'])] for x in g.values()]
        winners[key] = votes[0] if votes[0] == votes[1] else stable_choice(ids, [key[0], key[2]])
    return winners


def finals_manifest(prompts, configs, semifinal_results):
    winners = bracket_winners(semifinal_results)
    rows, missing = [], []
    for p, c in itertools.product(prompts, configs):
        a = winners.get((p['prompt_id'], c['config_id'], 'semi0'))
        b = winners.get((p['prompt_id'], c['config_id'], 'semi1'))
        if a is None or b is None:
            missing.append({'prompt_id': p['prompt_id'], 'config_id': c['config_id']})
        else:
            rows.extend(match_requests(p, c, a, b, 'final'))
    return rows, missing


def h3_outcomes(prompts, configs, results):
    winners = bracket_winners(results)
    out = []
    for p, c, n in itertools.product(prompts, configs, [1, 2, 4]):
        ids = p['candidate_subsets'][str(n)]
        chosen = ids[0] if n == 1 else winners.get((p['prompt_id'], c['config_id'], 'semi0' if n == 2 else 'final'))
        answers = {a['answer_id']: a for a in p['answers']}
        best = max(answers[i]['quality'] for i in ids)
        gold = stable_choice([i for i in ids if answers[i]['quality'] == best], [p['prompt_id'], 'gold', n])
        out.append({'prompt_id': p['prompt_id'], 'config_id': c['config_id'], 'n': n,
                    'source': p['source'], 'without_gpt4': p['without_gpt4'], 'selected_id': chosen,
                    'gold_id': gold, 'missing': chosen is None,
                    'format_drift': float(answers[chosen]['markdown']) - float(answers[gold]['markdown']) if chosen is not None else None,
                    'regret': best - answers[chosen]['quality'] if chosen is not None else None,
                    'selected_quality': answers[chosen]['quality'] if chosen is not None else None,
                    'gold_quality': best})
    return out


def human_agreement(rows, human_records):
    from .experiment import agreement
    gold = defaultdict(list)
    for r in human_records:
        gold[r['pair_id']].append(r['human_preference'])
    groups = defaultdict(dict)
    for r in rows:
        if r['condition'] == 'MM' and r['repeat'] == 0:
            groups[(r['config_id'], r['pair_id'])][r['order']] = r
    collected = defaultdict(lambda: {'gold': [], 'original': [], 'swap_average': []})
    for (cfg, pid), g in groups.items():
        original = g.get('AB', {})
        reversed_row = g.get('BA', {})
        a = original.get('verdict') if original.get('status') == 'ok' else None
        b = reversed_row.get('verdict') if reversed_row.get('status') == 'ok' else None
        averaged = a if a == b else 'TIE' if a is not None and b is not None else None
        for label in gold.get(pid, []):
            collected[cfg]['gold'].append(label)
            collected[cfg]['original'].append(a)
            collected[cfg]['swap_average'].append(averaged)
    return {cfg: {'human_annotations': len(v['gold']), 'original': agreement(v['gold'], v['original']),
                  'swap_average': agreement(v['gold'], v['swap_average'])} for cfg, v in collected.items()}
