"""Explicit post-hoc parsing sensitivity analysis; never edits the source run."""
import copy
import re
from pathlib import Path

from .core import digest, read_jsonl, save_json, write_jsonl
from .study import match_requests
from .uf_trial import collect_progress

POLICY = 'consistent_ab_anywhere_v1'


def recover_verdict(content, order='AB'):
    """Use only explicit A/B markers; reject conflicting markers and explicit ties."""
    labels = re.findall(r'\[\[([^\]]+)\]\]', content or '')
    if any(x.strip().upper() in {'TIE', 'C', 'A=B'} for x in labels):
        return None
    choices = {x for x in labels if x in {'A', 'B'}}
    if len(choices) != 1:
        return None
    return order['AB'.index(next(iter(choices)))]


def recovery_configs(configs):
    return [dict(c, parser_policy=POLICY) for c in configs]


def prepare_recovery(prompts, configs, source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Recovery needs a separate output folder')
    new_configs = recovery_configs(configs)
    signature = digest({'prompts': prompts, 'configs': new_configs})
    marker = destination / 'recovery_metadata.json'
    if marker.exists():
        import json
        if json.loads(marker.read_text())['signature'] != signature:
            raise ValueError('Recovery settings changed')
        collect_progress(prompts, new_configs, destination)
        return new_configs
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Destination must be empty for initial recovery')
    collect_progress(prompts, configs, source)  # Validate the original signatures first.
    by_prompt = {p['prompt_id']: p for p in prompts}
    audit, source_hashes = [], {}
    for model in dict.fromkeys(c['checkpoint'] for c in configs):
        group = [c for c in new_configs if c['checkpoint'] == model]
        by_config = {c['config_id']: c for c in group}
        path = source / model.split('/')[-1] / 'judge_results.jsonl'
        events = read_jsonl(path) if path.exists() else []
        source_hashes[model] = digest(events)
        latest = {e['request_id']: e for e in events}
        migrated = []
        for e in latest.values():
            c = by_config[e['config_id']]
            candidates = match_requests(by_prompt[e['prompt_id']], c, e['candidate_a'], e['candidate_b'], e['match_name'])
            request = next(r for r in candidates if r['order'] == e['order'])
            result = copy.deepcopy(e)
            result.update(request)
            result['run_signature'] = digest({'prompts': prompts, 'configs': group})
            result['source_request_id'] = e['request_id']
            result['original_status'] = e['status']
            result['original_verdict'] = e['verdict']
            result['parser_policy'] = POLICY
            # No guessing from prose/reasoning, no rescue of truncated/error outputs,
            # and no cherry-picking earlier attempts when the latest attempt fails.
            if e['status'] == 'invalid_verdict':
                verdict = recover_verdict(e.get('content'), e['order'])
                if verdict:
                    result.update(status='ok', verdict=verdict)
            migrated.append(result)
            audit.append({k: result[k] for k in ['request_id', 'source_request_id', 'config_id', 'attempt',
                         'original_status', 'original_verdict', 'status', 'verdict', 'parser_policy']})
        target = destination / model.split('/')[-1]
        target.mkdir(parents=True, exist_ok=True)
        write_jsonl(target / 'judge_results.jsonl', migrated)
    write_jsonl(destination / 'recovery_audit.jsonl', audit)
    save_json(destination / 'suite_metadata.json', {'signature': signature, 'configs': new_configs})
    save_json(marker, {'signature': signature, 'source': str(source), 'source_event_hashes': source_hashes,
        'policy': POLICY, 'note': 'POST-HOC SENSITIVITY ANALYSIS, not the original strict protocol. Latest saved attempt only. '
        'Same rule for all models/modes. Explicit consistent A/B markers anywhere; conflicting labels/ties rejected. '
        'Raw source logs preserved. Remaining finals require real inference; no winners fabricated. '
        'Original strict-parser retry selection is retained, so this is not a fresh run under the relaxed protocol.'})
    collect_progress(prompts, new_configs, destination, export=True)
    return new_configs
