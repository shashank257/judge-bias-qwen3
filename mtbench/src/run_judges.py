#!/usr/bin/env python3
"""Resumable MT-Bench pairwise judge harness for OpenAI-compatible chat APIs.

Python stdlib only. No network operations except POST to --base-url.
No calls are executed unless explicitly invoked with --execute.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CONDITIONS = [("plain", "plain"), ("markdown", "plain"),
              ("plain", "markdown"), ("markdown", "markdown")]
PROMPT_VERSION = "mtbench-format-pairwise-v1"
SYSTEM_PROMPT = (
    "You are an impartial evaluator. Compare the two assistant responses for how well "
    "they address the user's request. Evaluate factual correctness, helpfulness, "
    "relevance, completeness, and clarity. Do not reward a response solely for "
    "being displayed first. Respond with exactly one token: [[A]] or [[B]]. "
    "You must select one; ties are not allowed."
)
USER_TEMPLATE = (
    "Conversation context / user request:\n{question}\n\n"
    "Response A:\n{first}\n\nResponse B:\n{second}\n\n"
    "Which response is better? Output exactly [[A]] or [[B]]."
)


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def read_pairs(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def build_jobs(pairs, first_turn_only=True, limit=None):
    jobs = []
    filtered = 0
    seen = set()
    for rec in pairs:
        if first_turn_only and str(rec.get('turn', '1')) != '1':
            filtered += 1
            continue
        pair_id = rec['pair_id']
        if pair_id in seen:
            raise ValueError(f'Duplicate pair_id in input: {pair_id}')
        seen.add(pair_id)
        for afmt, bfmt in CONDITIONS:
            for order in ('AB', 'BA'):
                first_orig, second_orig = ('A', 'B') if order == 'AB' else ('B', 'A')
                fst = rec[f'answer_{first_orig.lower()}_{afmt if first_orig=="A" else bfmt}']
                snd = rec[f'answer_{second_orig.lower()}_{bfmt if second_orig=="B" else afmt}']
                base = {
                    'pair_id': pair_id, 'question_id': rec['question_id'],
                    'turn': rec.get('turn', ''), 'human_winner': rec.get('human_winner', ''),
                    'condition': f'A_{afmt}__B_{bfmt}', 'order': order,
                    'displayed_first': first_orig, 'displayed_second': second_orig,
                    'question_context': rec['question_context'],
                    'first_answer': fst, 'second_answer': snd,
                }
                jobs.append({**base, 'replicate': 0})
                # Exact duplicates of plain/plain per order for empirical noise floor.
                if afmt == 'plain' and bfmt == 'plain':
                    jobs.append({**base, 'replicate': 1})
        if limit is not None and len(seen) >= limit:
            break
    for job in jobs:
        job['job_id'] = sha({k: job[k] for k in ('pair_id','condition','order','replicate')})[:20]
    return jobs, filtered


def messages(job):
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': USER_TEMPLATE.format(question=job['question_context'],
                     first=job['first_answer'], second=job['second_answer'])},
    ]


def parse_verdict(content):
    text = content if isinstance(content, str) else ''
    match = re.fullmatch(r'\s*\[\[([AB])\]\]\s*', text)
    return match.group(1) if match else None


def post_json(url, key, body, timeout):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def existing_success_ids(path):
    successes = set()
    if path.exists():
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get('status') in ('ok', 'error'):
                            successes.add(record['job_id'])
                    except (ValueError, KeyError):
                        pass
    return successes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs', type=Path, required=True, help='mtbench_pilot_100.csv')
    ap.add_argument('--output', type=Path, default=Path('judge_logs.jsonl'))
    ap.add_argument('--model', required=True, help='Exact model name served by backend')
    ap.add_argument('--base-url', default=os.getenv('JUDGE_BASE_URL', 'http://localhost:8000/v1'))
    ap.add_argument('--api-key-env', default='JUDGE_API_KEY')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max-tokens', type=int, default=96)
    ap.add_argument('--timeout', type=int, default=120)
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--limit', type=int, help='Limit number of selected pairs (for smoke tests)')
    ap.add_argument('--include-multiturn', action='store_true', help='Risky: current pilot context may omit candidate-specific history')
    ap.add_argument('--extra-body-json', help='Optional server-specific request fields, e.g. Qwen thinking flags')
    ap.add_argument('--ollama-native', action='store_true', help='Use Ollama native /api/chat endpoint (reliable thinking controls)')
    ap.add_argument('--thinking', choices=('off','on'), default='off', help='Ollama native thinking mode (default off)')
    ap.add_argument('--execute', action='store_true', help='Actually send API requests (default: dry run)')
    args = ap.parse_args()
    if not args.pairs.exists():
        ap.error(f'Input CSV does not exist: {args.pairs}')
    if args.thinking == 'on' and not args.ollama_native:
        ap.error('--thinking on requires --ollama-native')
    if args.max_attempts < 1:
        ap.error('--max-attempts must be >= 1')
    pairs = read_pairs(args.pairs)
    jobs, excluded = build_jobs(pairs, first_turn_only=not args.include_multiturn, limit=args.limit)
    extra = json.loads(args.extra_body_json) if args.extra_body_json else {}
    if not isinstance(extra, dict):
        ap.error('--extra-body-json must be a JSON object')
    forbidden = {'model','messages','temperature','max_tokens','think','options','stream'} & extra.keys()
    if forbidden:
        ap.error(f'Extra body fields may not override: {sorted(forbidden)}')
    print(f'Input rows={len(pairs)}, excluded second-turn={excluded}, jobs={len(jobs)}, model={args.model}')
    print(f'Prompt version={PROMPT_VERSION}, prompt hash={sha([SYSTEM_PROMPT, USER_TEMPLATE])[:16]}')
    if not args.execute:
        print('DRY RUN: no API requests made. First job prompt:')
        if jobs:
            print(json.dumps(messages(jobs[0]), indent=2, ensure_ascii=False)[:3000])
        return
    if not jobs:
        raise SystemExit('No evaluation jobs selected.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_success_ids(args.output)
    key = os.getenv(args.api_key_env, '')
    endpoint = (args.base_url.rstrip('/').removesuffix('/v1') + '/api/chat'
                if args.ollama_native else args.base_url.rstrip('/') + '/chat/completions')
    for i, job in enumerate(jobs, 1):
        if job['job_id'] in done:
            continue
        if args.ollama_native:
            payload = {'model': args.model, 'messages': messages(job), 'stream': False,
                       'think': args.thinking == 'on',
                       'options': {'temperature': args.temperature, 'num_predict': args.max_tokens, "num_ctx": 8192},
                       **extra}
        else:
            payload = {'model':args.model, 'messages':messages(job),
                       'temperature':args.temperature, 'max_tokens':args.max_tokens, **extra}
        result = {k:v for k,v in job.items() if k not in ('question_context','first_answer','second_answer')}
        result.update({'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                       'model_requested':args.model, 'endpoint':endpoint,
                       'temperature':args.temperature, 'prompt_version':PROMPT_VERSION,
                       'system_prompt':SYSTEM_PROMPT, 'prompt_sha256':sha(payload['messages']),
                       'request_messages':payload['messages'], 'request_extra':extra,
                       'ollama_native': args.ollama_native,
                       'think': payload.get('think') if args.ollama_native else extra.get('think'),
                       'max_tokens':args.max_tokens, 'thinking_mode':args.thinking})
        error = None
        for attempt in range(1, args.max_attempts+1):
            try:
                response = post_json(endpoint, key, payload, args.timeout)
                if args.ollama_native:
                    raw_content = response.get('message', {}).get('content')
                    finish_reason = response.get('done_reason')
                else:
                    choice = response['choices'][0]
                    raw_content = choice.get('message', {}).get('content')
                    finish_reason = choice.get('finish_reason')
                # Retain generation metadata even when verdict parsing fails/truncates.
                result.update({'finish_reason':finish_reason,
                               'ollama_eval_count':response.get('eval_count'),
                               'ollama_thinking_tokens':response.get('thinking_eval_count'),
                               'raw_response':response})
                verdict = parse_verdict(raw_content)
                if not verdict:
                    raise ValueError(f'Invalid verdict format: {raw_content!r}')
                result.update({'status':'ok', 'raw_response':response,
                               'finish_reason': finish_reason,
                               'judge_verdict_displayed':verdict,
                               'judge_winner_original':job['displayed_first'] if verdict=='A' else job['displayed_second'],
                               'model_returned':response.get('model'), 'usage':response.get('usage'),
                               'ollama_eval_count':response.get('eval_count'),
                               'ollama_thinking_tokens':response.get('thinking_eval_count'),
                               'attempts':attempt})
                error = None
                break
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
                error = f'{type(exc).__name__}: {exc}'
                if attempt < args.max_attempts:
                    time.sleep(min(2**(attempt-1), 8) + random.random()*0.1)
        if error:
            result.update({'status':'error', 'error':error, 'attempts':args.max_attempts})
        with args.output.open('a', encoding='utf-8') as out:
            out.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f'[{i}/{len(jobs)}] {job["pair_id"]} {job["condition"]} {job["order"]} repeat={job["replicate"]}: {result["status"]}', flush=True)
    print('Done. Successful jobs will be skipped when this command is rerun.')

if __name__ == '__main__':
    main()
