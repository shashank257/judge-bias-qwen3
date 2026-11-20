"""Public catalog checks and resumable, explicit opt-in API execution."""
import json
import os
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from .core import digest, save_json, read_jsonl
from .experiment import parse_verdict

SIZES = ['0.6B','1.7B','4B','8B']


def catalog_snapshot(out):
    import requests
    out = Path(out)
    snapshot = {'checked_utc':datetime.now(timezone.utc).isoformat(), 'openrouter':{}, 'huggingface':{}}
    try:
        r = requests.get('https://openrouter.ai/api/v1/models', timeout=30)
        r.raise_for_status()
        models = r.json()['data']
        save_json(out / 'openrouter_catalog.json', models)
        for size in SIZES:
            expected = 'qwen/qwen3-' + size.lower()
            matches = [m for m in models if m['id'] == expected]
            snapshot['openrouter'][size] = {'expected_id':expected, 'listed':bool(matches), 'models':matches}
    except (requests.RequestException, ValueError, KeyError) as e:
        snapshot['openrouter']['error'] = str(e)
    for size in SIZES:
        try:
            url = f'https://huggingface.co/api/models/Qwen/Qwen3-{size}?expand=inferenceProviderMapping'
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            snapshot['huggingface'][size] = r.json()
        except (requests.RequestException, ValueError) as e:
            snapshot['huggingface'][size] = {'error':str(e)}
    save_json(out / 'catalog_check.json', snapshot)
    return snapshot


def validate_config(config, pilot=False):
    required = ['config_id','model_id','checkpoint','base_url','key_env','provider','thinking','sampling','extra_body']
    for key in required:
        if key not in config:
            raise ValueError(f'Missing {key}')
    if config['checkpoint'] not in [f'Qwen/Qwen3-{s}' for s in SIZES]:
        raise ValueError('Only original requested hybrid checkpoints allowed. Revise contract explicitly for other models.')
    if not config.get('availability_verified'):
        raise ValueError('Confirm exact model and endpoint availability before paid calls.')
    if pilot and not config.get('controls_verified'):
        raise ValueError('Complete and record thinking-control compatibility checks before the pilot.')
    if config['provider'] == 'openrouter':
        route = config['extra_body'].get('provider', {})
        if not route.get('only') or route.get('allow_fallbacks') is not False or route.get('require_parameters') is not True:
            raise ValueError('Pin provider only; set allow_fallbacks=false and require_parameters=true.')
    reserved = {'model','messages','stream','seed'}
    if reserved.intersection(config['extra_body']) or reserved.intersection(config['sampling']):
        raise ValueError('Do not override model/messages/stream or fix repeat seeds in sampling/extra_body.')
    if not isinstance(config['sampling'].get('max_tokens'), int) or config['sampling']['max_tokens'] <= 0:
        raise ValueError('Positive total max_tokens required.')


def run_manifest(manifest, configs, log_path, execute=False, max_calls=8, pilot=False, timeout=180):
    """Sequential by design. max_calls includes retries and prior starts in this log.

    Each repeat has a distinct request ID and gets a fresh POST, without a fixed API seed.
    Unknown outcomes after interruption are not automatically resent. Inspect them first.
    """
    import requests
    by_id = {c['config_id']:c for c in configs}
    if len(by_id) != len(configs):
        raise ValueError('Duplicate configuration IDs')
    if len({r['request_id'] for r in manifest}) != len(manifest):
        raise ValueError('Duplicate request IDs')
    for r in manifest:
        c = by_id[r['config_id']]
        if r['config_hash'] != digest(c):
            raise ValueError('Configuration changed after manifest creation; rebuild manifest.')
    if not execute:
        return {'dry_run':True, 'unique_requests':len(manifest), 'max_calls':max_calls,
                'note':'No network generations. Set execute=True only after reviewing configurations and account spending limit.'}
    for c in configs:
        validate_config(c, pilot=pilot)
        if not os.environ.get(c['key_env']):
            raise ValueError(f'Set {c["key_env"]} in Colab Secrets/environment; never put keys into files.')
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(path) if path.exists() else []
    signature = digest([r['request_id'] for r in manifest])
    if any(e.get('manifest_hash') != signature for e in existing):
        raise ValueError('Use a separate log path for a different manifest.')
    starts = [e for e in existing if e['event'] == 'start']
    ends = {e['attempt_id'] for e in existing if e['event'] == 'result'}
    uncertain = {e['request_id'] for e in starts if e['attempt_id'] not in ends}
    results = [e for e in existing if e['event'] == 'result']
    done = {e['request_id'] for e in results if not e.get('retry_pending', e.get('retry_scheduled', False))}
    calls = len(starts)

    def log(event):
        event['manifest_hash'] = signature
        with path.open('a') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())

    for row in manifest:
        if row['request_id'] in done or row['request_id'] in uncertain:
            continue
        c = by_id[row['config_id']]
        payload = dict(model=c['model_id'], messages=row['messages'], stream=False, **c['sampling'], **c['extra_body'])
        history = [e for e in results if e['request_id'] == row['request_id']]
        invalid_retries = sum(e['status'] == 'invalid_verdict' for e in history)
        http_retries = sum(e['status'] == 'http_error' for e in history)
        for attempt in range(len(history), 4):
            if calls >= max_calls:
                return {'calls_total':calls, 'finished_requests':len(done), 'uncertain_requests':sorted(uncertain), 'stopped':'call_limit'}
            aid = str(uuid.uuid4())
            base = {'request_id':row['request_id'], 'config_id':c['config_id'], 'attempt_id':aid, 'attempt':attempt,
                    'utc':datetime.now(timezone.utc).isoformat()}
            log(dict(base, event='start', payload_hash=digest(payload)))
            calls += 1
            began = time.perf_counter()
            retryable = False
            result = dict(base, event='result', verdict=None)
            try:
                response = requests.post(c['base_url'].rstrip('/') + '/chat/completions',
                                         headers={'Authorization':'Bearer ' + os.environ[c['key_env']]},
                                         json=payload, timeout=timeout)
                result['http_status'] = response.status_code
                try:
                    body = response.json()
                except ValueError:
                    body = {'non_json_response':response.text[:4000]}
                result['raw_response'] = body
                if response.status_code >= 400:
                    result['status'] = 'http_error'
                    retryable = (response.status_code == 429 or response.status_code >= 500) and http_retries < 2
                    if retryable:
                        http_retries += 1
                elif body.get('error') or not body.get('choices'):
                    result['status'] = 'response_error'
                else:
                    choice = body['choices'][0]
                    message = choice.get('message', {})
                    result.update(finish_reason=choice.get('finish_reason'), usage=body.get('usage', {}),
                                  returned_model=body.get('model'), serving_provider=body.get('provider'),
                                  system_fingerprint=body.get('system_fingerprint'),
                                  reasoning_visible=bool(message.get('reasoning') or message.get('reasoning_content') or message.get('reasoning_details')))
                    result['verdict'] = parse_verdict(message.get('content'), row['order'], c.get('verdict_policy', 'strict_tie'))
                    result['status'] = 'truncated' if choice.get('finish_reason') == 'length' else 'ok' if result['verdict'] else 'invalid_verdict'
                    if result['status'] != 'ok':
                        result['verdict'] = None
                    if result['status'] == 'invalid_verdict' and invalid_retries < c.get('invalid_verdict_retries', 0):
                        invalid_retries += 1
                        retryable = True
            except requests.RequestException as e:
                # Timeout may have been billed. Do not silently duplicate that generation.
                result.update(status='transport_uncertain', error=type(e).__name__)
            result['latency_s'] = time.perf_counter() - began
            result['retry_pending'] = retryable and attempt < 3
            result['retry_scheduled'] = result['retry_pending'] and calls < max_calls
            result['invalid_retries_used'] = invalid_retries
            log(result)
            if result['retry_scheduled']:
                time.sleep(min(2 ** (attempt+1), 8))
            elif not result['retry_pending']:
                done.add(row['request_id'])
                break
    return {'calls_total':calls, 'finished_requests':len(done), 'uncertain_requests':sorted(uncertain)}


def summarize_run(log_path):
    import statistics
    events = read_jsonl(log_path)
    final, grouped = {}, defaultdict(list)
    attempts = [e for e in events if e['event'] == 'result']
    for e in attempts:
        final[e['request_id']] = e
    for r in final.values():
        grouped[r['config_id']].append(r)
    summary = {}
    for cfg, rows in grouped.items():
        usage_rows = [r for r in attempts if r['config_id'] == cfg]
        costs = [r.get('usage', {}).get('cost') for r in usage_rows]
        summary[cfg] = {'finished_requests':len(rows), 'statuses':dict(Counter(r['status'] for r in rows)),
                        'median_latency_s':statistics.median(r['latency_s'] for r in rows),
                        'prompt_tokens_reported':sum(r.get('usage', {}).get('prompt_tokens',0) or 0 for r in usage_rows),
                        'completion_tokens_reported':sum(r.get('usage', {}).get('completion_tokens',0) or 0 for r in usage_rows),
                        'reported_cost_sum':sum(c for c in costs if isinstance(c,(int,float))),
                        'attempts_without_cost':sum(not isinstance(c,(int,float)) for c in costs),
                        'cost_complete':all(isinstance(c,(int,float)) for c in costs)}
    return summary
