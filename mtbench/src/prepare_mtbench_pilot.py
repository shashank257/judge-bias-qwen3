#!/usr/bin/env python3
"""Prepare a reproducible MT-Bench 100-pair pilot and 8-format evaluations.

Run after: pip install datasets
Usage: python prepare_mtbench_pilot.py

Outputs:
  mtbench_all_unique_pairs.csv  - eligible grouped answer pairs with human votes
  mtbench_pilot_100.csv         - deterministic 100-pair pilot
  mtbench_pilot_conditions.csv  - 4 formats x 2 orders per selected pair
  mtbench_pilot_summary.json    - filters and counts

No judge calls are made. Human votes are the dataset labels, not judge decisions.
"""
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DATASET_NAME = 'lmsys/mt_bench_human_judgments'
SEED = 2612026
N_PILOT = 100
OUT = Path(__file__).resolve().parent


def as_messages(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    return [dict(item) for item in value]


def pair_for_turn(record):
    # 'turn' is the numbered assistant answer under evaluation (1 or 2).
    a, b = as_messages(record['conversation_a']), as_messages(record['conversation_b'])
    t = int(record['turn'])
    a_assist = [m for m in a if m.get('role') == 'assistant']
    b_assist = [m for m in b if m.get('role') == 'assistant']
    if not (1 <= t <= len(a_assist) and 1 <= t <= len(b_assist)):
        return None
    # For multi-turn records, use the conversation prefix through the corresponding
    # user turn rather than misrepresenting the second-turn question as standalone.
    a_seen = 0
    query_parts = []
    for m in a:
        if m.get('role') == 'assistant':
            a_seen += 1
            if a_seen >= t:
                break
        query_parts.append(f"{m.get('role','unknown').upper()}: {m.get('content','')}")
    prompt = '\n\n'.join(query_parts)
    text_a = str(a_assist[t-1].get('content', ''))
    text_b = str(b_assist[t-1].get('content', ''))
    return prompt, text_a, text_b


# Conservative filter: the primary analysis excludes markup/list/code/table/formats
# where plain-text conversion could alter meaning or response structure.
def exclude_reason(text):
    if not text or len(text.strip()) < 40:
        return 'too_short'
    if len(text) > 3500:
        return 'too_long'
    if '\n' in text or '\r' in text:
        return 'multiline_or_list'
    if re.search(r'```|`|\*\*|__|\[[^\]]+\]\([^)]*\)|<[^>]+>|^\s*#|\|', text):
        return 'existing_markup_or_code'
    if re.search(r'^\s*(?:[-*+]\s|\d+[.)]\s)', text):
        return 'list_or_procedure'
    if re.search(r'\b(?:first|second|third|step\s+[1-9])\s*:', text, re.I):
        return 'possible_procedure'
    return None


def to_markdown(plain):
    # Insert bold markers around the first 2-4 complete words, preserving exact text.
    matches = list(re.finditer(r'\b[A-Za-z][A-Za-z\'-]*\b', plain))
    if len(matches) < 4:
        return None
    start, end = matches[0].start(), matches[3].end()
    return plain[:start] + '**' + plain[start:end] + '**' + plain[end:]


def undo_markdown(text):
    return text.replace('**', '')


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit('Install dependency first: pip install datasets') from exc
    rows = load_dataset(DATASET_NAME, split='human')
    tally = Counter()
    groups = defaultdict(lambda: Counter())
    metadata = {}
    for rec in rows:
        tally['raw_human_judgments'] += 1
        winner = rec.get('winner')
        if winner not in ('model_a', 'model_b'):
            tally['removed_ties'] += 1
            continue
        parsed = pair_for_turn(rec)
        if parsed is None:
            tally['invalid_conversation'] += 1
            continue
        prompt, a, b = parsed
        if not prompt.strip():
            tally['empty_prompt'] += 1
            continue
        reason = exclude_reason(a) or exclude_reason(b)
        if reason:
            tally[f'removed_{reason}'] += 1
            continue
        a_md, b_md = to_markdown(a), to_markdown(b)
        if not a_md or not b_md or undo_markdown(a_md) != a or undo_markdown(b_md) != b:
            tally['failed_markup_equivalence'] += 1
            continue
        key_payload = [int(rec['question_id']), int(rec['turn']), str(rec['model_a']), str(rec['model_b']), prompt, a, b]
        key = hashlib.sha256(json.dumps(key_payload, ensure_ascii=False).encode()).hexdigest()[:16]
        groups[key][winner] += 1
        metadata[key] = dict(pair_id=key, question_id=int(rec['question_id']), turn=int(rec['turn']),
                             model_a=str(rec['model_a']), model_b=str(rec['model_b']),
                             question_context=prompt, answer_a_original=a, answer_b_original=b,
                             answer_a_plain=a, answer_a_markdown=a_md,
                             answer_b_plain=b, answer_b_markdown=b_md)
    prepared = []
    for key, votes in groups.items():
        if votes['model_a'] == votes['model_b']:
            tally['removed_tied_majority'] += 1
            continue
        human_winner = 'A' if votes['model_a'] > votes['model_b'] else 'B'
        prepared.append({**metadata[key], 'human_winner':human_winner,
                         'votes_a': votes['model_a'], 'votes_b':votes['model_b'],
                         'human_judgment_count':sum(votes.values()), 'source': 'mt_bench_human_judgments'})
    prepared.sort(key=lambda x: x['pair_id'])
    cols = list(prepared[0]) if prepared else list(dict.fromkeys(list(metadata[next(iter(metadata))]) + ['human_winner','votes_a','votes_b','human_judgment_count','source'])) if metadata else []
    if not prepared:
        raise RuntimeError('No eligible rows after filtering; inspect the filters and source schema.')
    write_csv(OUT / 'mtbench_all_unique_pairs.csv', prepared, cols)
    if len(prepared) < N_PILOT:
        raise RuntimeError(f'Only {len(prepared)} eligible pairs; lower constraints or expand dataset with a documented protocol.')
    # Spread sample across question_id clusters where feasible, to avoid an
    # overconcentration from repeatedly judged versions of the same question.
    by_question = defaultdict(list)
    for item in prepared:
        by_question[item['question_id']].append(item)
    rng = random.Random(SEED)
    buckets = list(by_question)
    rng.shuffle(buckets)
    for qid in buckets:
        rng.shuffle(by_question[qid])
    pilot = []
    while len(pilot) < N_PILOT:
        added = False
        for qid in buckets:
            if by_question[qid]:
                pilot.append(by_question[qid].pop())
                added = True
                if len(pilot) == N_PILOT:
                    break
        if not added:
            break
    write_csv(OUT / 'mtbench_pilot_100.csv', pilot, cols)
    conditions = []
    for rec in pilot:
        for a_fmt,b_fmt in [('plain','plain'),('markdown','plain'),('plain','markdown'),('markdown','markdown')]:
            for order in ['AB','BA']:
                a = rec[f'answer_a_{a_fmt}']
                b = rec[f'answer_b_{b_fmt}']
                conditions.append(dict(pair_id=rec['pair_id'], question_id=rec['question_id'],
                    question_context=rec['question_context'], condition=f'A_{a_fmt}__B_{b_fmt}',
                    order=order, displayed_first='A' if order=='AB' else 'B',
                    first_answer=a if order=='AB' else b,
                    displayed_second='B' if order=='AB' else 'A',
                    second_answer=b if order=='AB' else a,
                    human_winner=rec['human_winner'], judge_winner='', judge_model='', prompt_version=''))
    write_csv(OUT / 'mtbench_pilot_conditions.csv', conditions, list(conditions[0]))
    summary = {'dataset':DATASET_NAME, 'split':'human','seed':SEED,'pilot_pairs':len(pilot),
               'unique_eligible_pairs':len(prepared),'distinct_pilot_question_ids':len({r['question_id'] for r in pilot}),
               'format_order_conditions':len(conditions),'counts':dict(tally),
               'caution': 'Manual audit of generated variants is required before model evaluation; markup added to first 4 words only.',
               'note': 'Data may have fewer than 100 distinct question IDs; bootstrap should resample by question_id.'}
    (OUT/'mtbench_pilot_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
