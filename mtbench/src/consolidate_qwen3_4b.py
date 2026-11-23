#!/usr/bin/env python3
"""Consolidate Qwen3-4B thinking-on judgments without changing source logs."""
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import run_judges as harness

LOG_DIR = Path('logs')
SOURCES = [
    ('greedy_4096', LOG_DIR / 'qwen3_4b_matched31_on_4096.jsonl', 4096, 0.0),
    ('greedy_8192_recovery', LOG_DIR / 'qwen3_4b_matched31_on_8192_recovery.jsonl', 8192, 0.0),
    ('sampled_8192_final', LOG_DIR / 'qwen3_4b_matched31_on_sampled_final_recovery.jsonl', 8192, 0.6),
]
OUT = LOG_DIR / 'qwen3_4b_matched31_on_consolidated.jsonl'
AUDIT = LOG_DIR / 'qwen3_4b_matched31_on_consolidation_audit.json'
CSV_PATH = '../mtbench_pilot_outputs/mtbench_matched_31.csv'

pairs = harness.read_pairs(CSV_PATH)
jobs, excluded = harness.build_jobs(pairs, first_turn_only=True, limit=None)
by_id = {j['job_id']: j for j in jobs}
assert len(jobs) == len(by_id) == 310, 'Expected 310 unique jobs'

candidates = {}
audit = {'sources': {}, 'unknown_job_ids': [], 'invalid_ok_records': [], 'duplicate_valid_job_ids': [], 'conflicting_verdicts': []}
for source_index, (source, path, budget, temp) in enumerate(SOURCES):
    if not path.exists():
        raise FileNotFoundError(path)
    counters = Counter()
    with path.open(encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            counters['records'] += 1
            jid = r.get('job_id')
            if jid not in by_id:
                audit['unknown_job_ids'].append({'source': source, 'line': lineno, 'job_id': jid})
                continue
            if r.get('status') != 'ok':
                counters['unsuccessful'] += 1
                continue
            # Older harness logs can store parsed winners without response_text.
            # Accept only an explicit A/B winner, mapped using canonical job order.
            answer = (r.get('response_text') or '').strip()
            parsed_from_text = harness.parse_verdict(answer) if answer else None
            recorded_displayed = r.get('judge_winner_displayed')
            recorded_original = r.get('judge_winner_original')
            first = by_id[jid]['displayed_first']
            original_to_displayed = (
                recorded_original if first == 'A'
                else ('B' if recorded_original == 'A' else 'A')
            ) if recorded_original in ('A', 'B') else None
            claims = [v for v in (parsed_from_text, recorded_displayed, original_to_displayed) if v in ('A', 'B')]
            if answer and parsed_from_text not in ('A', 'B'):
                audit['invalid_ok_records'].append({'source': source, 'line': lineno, 'job_id': jid, 'reason': 'nonempty invalid response_text', 'answer': answer[:100]})
                continue
            if not claims or len(set(claims)) != 1:
                audit['invalid_ok_records'].append({'source': source, 'line': lineno, 'job_id': jid, 'reason': 'missing or conflicting recorded winners', 'response_text': answer[:100], 'displayed': recorded_displayed, 'original': recorded_original})
                continue
            parsed = claims[0]
            candidates.setdefault(jid, []).append((source_index, lineno, source, budget, temp, r, parsed))
            counters['valid'] += 1
    audit['sources'][source] = {'path': str(path), **dict(counters)}

consolidated = []
for job in jobs:
    jid = job['job_id']
    entries = candidates.get(jid, [])
    if not entries:
        continue
    entries.sort(key=lambda x: (x[0], x[1]))  # earliest valid in canonical source order
    if len(entries) > 1:
        audit['duplicate_valid_job_ids'].append({'job_id': jid, 'count': len(entries)})
    verdicts = {e[6] for e in entries}
    if len(verdicts) > 1:
        audit['conflicting_verdicts'].append({'job_id': jid, 'verdicts': sorted(verdicts)})
    _, lineno, source, budget, temp, rec, displayed = entries[0]
    # Verify using canonical job definition, not assumptions about display metadata.
    first = job['displayed_first']
    assert first in ('A', 'B'), (jid, first)
    original = first if displayed == 'A' else ('B' if first == 'A' else 'A')
    result = dict(rec)
    result.update({
        'job_id': jid,
        'status': 'ok',
        'judge_winner_displayed': displayed,
        'judge_winner_original': original,
        'consolidation_source': source,
        'consolidation_source_line': lineno,
        'consolidation_generation_budget': budget,
        'consolidation_temperature': temp,
        'consolidation_protocol': 'thinking_on_adaptive_budget_with_final_sampled_recovery',
    })
    consolidated.append(result)

audit.update({
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'expected_jobs': len(jobs),
    'excluded_second_turn': excluded,
    'selected_valid_jobs': len(consolidated),
    'missing_job_ids': [j['job_id'] for j in jobs if j['job_id'] not in candidates],
    'selected_by_source': dict(Counter(r['consolidation_source'] for r in consolidated)),
    'output': str(OUT),
})
if audit['missing_job_ids'] or audit['unknown_job_ids'] or audit['invalid_ok_records']:
    raise RuntimeError('Validation failed; no output written. ' + json.dumps({k: audit[k] for k in ('missing_job_ids', 'unknown_job_ids', 'invalid_ok_records')}, indent=2))

# Write new output atomically to avoid half-written consolidated files.
tmp = OUT.with_suffix('.jsonl.tmp')
with tmp.open('w', encoding='utf-8') as f:
    for row in consolidated:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
tmp.replace(OUT)
AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('Saved:', OUT)
print('Valid unique jobs:', len(consolidated), '/310')
print('Selected by source:', audit['selected_by_source'])
print('Conflicting successful verdicts:', len(audit['conflicting_verdicts']))
print('Audit:', AUDIT)
