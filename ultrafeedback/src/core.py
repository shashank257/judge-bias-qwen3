"""Deterministic data preparation. No model calls or credentials required."""
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

SEED = 261
VERSION = '1.1.0'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sample(rows, n, seed=SEED):
    if len(rows) < n:
        raise ValueError(f'Need {n} eligible records; only {len(rows)} remain. Do not lower target silently.')
    return random.Random(seed).sample(sorted(rows, key=lambda r: r['pair_id'] if 'pair_id' in r else r['prompt_id']), n)


def words(text):
    return re.findall(r"\w+(?:['’]\w+)*", text, re.UNICODE)


def forbidden_content(text):
    # Conservative: includes indented code and HTML tables/pre blocks.
    return bool(re.search(r'(?m)^\s*(`{3,}|~{3,})|^(?: {4}|\t)\S|^\s*\|?.*\|.*\n\s*\|?\s*:?-{3,}', text)
                or re.search(r'<(?:table|pre|code)\b', text, re.I))


def strip_markdown(text):
    """Delete supported marker spans only; retain every other byte and newline."""
    spans = []
    for m in re.finditer(r'(?m)^( {0,3})(#{1,6})[ \t]+', text):
        spans.append((m.start(2), m.end()))
        line_end = text.find('\n', m.end())
        line_end = len(text) if line_end < 0 else line_end
        closing = re.search(r'[ \t]+(#+)[ \t]*$', text[m.end():line_end])
        if closing:
            spans.append((m.end() + closing.start(1), m.end() + closing.end(1)))
    for m in re.finditer(r'(?m)^( {0,3})(?:[-+*]|\d+[.)])[ \t]+(?=\S)', text):
        spans.append((m.start() + len(m.group(1)), m.end()))
    for m in re.finditer(r'(?<![\\\w*])\*\*(?=\S)(.+?)(?<=\S)\*\*(?!\*)|(?<![\\\w_])__(?=\S)(.+?)(?<=\S)__(?![\w_])', text):
        spans.extend([(m.start(), m.start() + 2), (m.end() - 2, m.end())])
    spans = sorted(set(spans))
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError('Overlapping markdown markers require manual review')
    parts, cursor = [], 0
    for start, end in spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return ''.join(parts), spans


def validate_strip(original, plain, spans):
    # Independent lexical check: only list numbering and bold underscores can be non-content words.
    lexical = re.sub(r'(?m)^ {0,3}\d+[.)][ \t]+', '', original)
    lexical = re.sub(r'(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)', r'\1', lexical)
    lexical_ok = words(lexical) == words(plain)
    removed = {i for a, b in spans for i in range(a, b)}
    exact_ok = plain == ''.join(c for i, c in enumerate(original) if i not in removed)
    return lexical_ok and exact_ok and original.count('\n') == plain.count('\n')


def unsupported_markdown(plain):
    # Conservative format-pool restriction: avoid calling partially stripped text "plain".
    return bool(re.search(r'`|\*|(?<!\w)_(?=\S)|\[[^\]]*\]\(|!\[|(?m:^\s*>|^\s*#{1,6}\s|^\s*[-=]{3,}\s*$)|</?\w+[^>]*>', plain))


def markdown_info(text):
    try:
        plain, spans = strip_markdown(text)
        valid = validate_strip(text, plain, spans)
    except ValueError:
        plain, spans, valid = text, [], False
    return {'plain': plain, 'has_markdown': bool(spans), 'strip_valid': valid,
            'format_supported': valid and not unsupported_markdown(plain), 'removed_spans': spans}


def normalized_prompt(text):
    # No lowercasing: case can matter. Normalize whitespace only for clustering.
    return ' '.join(text.split())


def prepare_arena(rows, source='arena_2023'):
    kept, failures, excluded, seen, stages = [], [], Counter(), set(), Counter()
    for index, row in enumerate(rows):
        stages['input'] += 1
        reason = None
        if str(row.get('language', '')).lower() not in {'english', 'en'}:
            reason = 'non_english_or_missing_language'
        else:
            stages['english'] += 1
            a, b = row.get('conversation_a'), row.get('conversation_b')
            valid = all(isinstance(c, list) and len(c) == 2 and
                        [x.get('role') for x in c] == ['user', 'assistant'] and
                        all(isinstance(x.get('content'), str) for x in c) for c in [a, b])
            if not valid or row.get('turn', 1) != 1:
                reason = 'not_single_turn_or_invalid_structure'
            elif not a[0]['content'].strip() or a[0]['content'] != b[0]['content']:
                reason = 'missing_or_mismatched_prompt'
            else:
                stages['single_turn_aligned'] += 1
                if row.get('winner') not in {'model_a', 'model_b'}:
                    reason = 'tie_or_invalid_label'
                else:
                    stages['decisive_label'] += 1
                    question, aa, bb = a[0]['content'], a[1]['content'], b[1]['content']
                    la, lb = len(words(aa)), len(words(bb))
                    if not min(la, lb):
                        reason = 'empty_answer'
                    else:
                        stages['nonempty_answers'] += 1
                        if max(la, lb) / min(la, lb) > 2:
                            reason = 'length_ratio_over_2'
                        else:
                            stages['length_ratio'] += 1
                            if forbidden_content(aa) or forbidden_content(bb):
                                reason = 'table_or_code'
                            else:
                                stages['content_rule'] += 1
                                # Duplicate votes on exactly the same presentation are excluded; prompt repeats remain.
                                key = digest([question, aa, bb])
                                if key in seen:
                                    reason = 'duplicate_pair'
                                else:
                                    seen.add(key)
                                    stages['deduplicated'] += 1
                                    ma, mb = markdown_info(aa), markdown_info(bb)
                                    record = dict(pair_id=key, prompt_id=digest(normalized_prompt(question)),
                                                  source=source, source_row=index, source_question_id=row.get('question_id'),
                                                  question=question, answer_a=aa, answer_b=bb,
                                                  human_preference='A' if row['winner'] == 'model_a' else 'B',
                                                  length_a=la, length_b=lb, model_a=row.get('model_a'), model_b=row.get('model_b'),
                                                  markdown_a=ma['has_markdown'], markdown_b=mb['has_markdown'],
                                                  plain_a=ma['plain'], plain_b=mb['plain'],
                                                  format_eligible=all(m['has_markdown'] and m['format_supported'] for m in [ma, mb]))
                                    if not ma['strip_valid'] or not mb['strip_valid']:
                                        failures.append(dict(pair_id=key, answer_a=aa, answer_b=bb, a=ma, b=mb))
                                        excluded['strip_validation_failure'] += 1
                                    else:
                                        stages['transformation_valid'] += 1
                                        kept.append(record)
        if reason:
            excluded[reason] += 1
    ordered = ['input', 'english', 'single_turn_aligned', 'decisive_label', 'nonempty_answers', 'length_ratio', 'content_rule', 'deduplicated', 'transformation_valid']
    counts, previous = [], stages['input']
    for step in ordered:
        count = stages[step]
        counts.append({'step': step, 'retained': count, 'excluded_here': previous - count})
        previous = count
    prompt_counts = Counter(r['prompt_id'] for r in kept)
    report = {'version': VERSION, 'source': source, 'steps': counts, 'excluded_reasons': dict(excluded),
              'format_eligible': sum(r['format_eligible'] for r in kept), 'strip_failures': len(failures),
              'unique_prompts': len(prompt_counts), 'repeated_prompt_clusters': sum(n > 1 for n in prompt_counts.values())}
    return kept, report, failures


def create_audit(rows, path, n=50):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} already exists; preserve completed human work.')
    chosen = sample(rows, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pair_id', 'question', 'answer_a', 'plain_a', 'answer_b', 'plain_b',
                                              'meaning_ok', 'wording_ok', 'list_structure_ok', 'reviewer', 'notes'])
        writer.writeheader()
        for r in chosen:
            writer.writerow({k: r[k] for k in ['pair_id', 'question', 'answer_a', 'plain_a', 'answer_b', 'plain_b']})


def freeze_pools(rows, audit_path, out, n=1000):
    with Path(audit_path).open() as f:
        audit = list(csv.DictReader(f))
    ids = {r['pair_id'] for r in rows if r['format_eligible']}
    if len(audit) != 50 or len({r['pair_id'] for r in audit}) != 50:
        raise ValueError('Exactly 50 distinct audited pairs required.')
    if not {r['pair_id'] for r in audit}.issubset(ids):
        raise ValueError('Audit does not match this candidate pool.')
    if any(not r['reviewer'].strip() or any(r[k].strip().lower() not in {'yes','no'} for k in ['meaning_ok','wording_ok','list_structure_ok']) for r in audit):
        raise ValueError('Complete manual audit with yes/no and reviewer before freezing.')
    rejected = {r['pair_id'] for r in audit if any(r[k].strip().lower() == 'no' for k in ['meaning_ok','wording_ok','list_structure_ok'])}
    eligible = [r for r in rows if r['pair_id'] not in rejected]
    pools = {'position': sample(eligible, n), 'format': sample([r for r in eligible if r['format_eligible']], n)}
    out = Path(out)
    if (out / 'freeze.json').exists():
        raise FileExistsError('Use a new output directory for a new freeze version.')
    for key, records in pools.items():
        write_jsonl(out / f'{key}.jsonl', records)
    save_json(out / 'freeze.json', {'version': VERSION, 'seed': SEED, 'candidate_hash': digest(rows),
                                  'audit_hash': digest(audit), 'rejected_audit_pairs': sorted(rejected),
                                  'pool_hashes': {k: digest(v) for k, v in pools.items()}})
    return pools


def prepare_ultrafeedback(rows, n=500):
    eligible, excluded, seen = [], Counter(), set()
    aspects = ['instruction_following', 'truthfulness', 'honesty', 'helpfulness']
    for row in rows:
        source = row.get('source')
        if source not in {'sharegpt', 'ultrachat', 'evol_instruct'}:
            excluded['source'] += 1
            continue
        question, completions = row.get('instruction'), row.get('completions')
        if not isinstance(question, str) or not question.strip() or not isinstance(completions, list) or len(completions) != 4:
            excluded['question_or_four_answers'] += 1
            continue
        answers = []
        for i, c in enumerate(completions):
            try:
                ratings = [float(c['annotations'][a]['Rating']) for a in aspects]
                if not all(1 <= r <= 5 for r in ratings) or not c['response'].strip() or not c['model']:
                    raise ValueError('Unusable rating or answer')
                answers.append({'answer_id': str(i), 'text': c['response'], 'model': c['model'],
                                'ratings': dict(zip(aspects, ratings)), 'quality': sum(ratings) / 4,
                                'markdown': markdown_info(c['response'])['has_markdown']})
            except (KeyError, TypeError, ValueError, AttributeError):
                break
        if len(answers) != 4:
            excluded['missing_or_invalid_ratings'] += 1
            continue
        pid = digest(normalized_prompt(question))
        if pid in seen:
            excluded['duplicate_prompt'] += 1
            continue
        seen.add(pid)
        eligible.append({'prompt_id': pid, 'source': source, 'question': question, 'answers': answers,
                         'without_gpt4': not any(re.search(r'gpt[-_ ]?4', a['model'], re.I) for a in answers)})
    selected = sample(eligible, n)
    for row in selected:
        perm = random.Random(int(row['prompt_id'][:16], 16) + SEED).sample(['0','1','2','3'], 4)
        row['candidate_subsets'] = {str(k): perm[:k] for k in [1,2,4]}
        row['brackets'] = {'1': [], '2': [[perm[0], perm[1]]],
                           '4': [[perm[0], perm[1]], [perm[2], perm[3]], ['winner_match_0', 'winner_match_1']]}
    return selected, {'eligible': len(eligible), 'selected': len(selected), 'excluded': dict(excluded),
                      'sources': dict(Counter(r['source'] for r in selected)),
                      'without_gpt4': sum(r['without_gpt4'] for r in selected)}


def prepare_mtbench(rows):
    kept, excluded = [], Counter()
    for idx, r in enumerate(rows):
        turn = r.get('turn')
        a, b = r.get('conversation_a'), r.get('conversation_b')
        if turn not in [1, 2] or not all(isinstance(c, list) and len(c) >= 2 * turn for c in [a,b]):
            excluded['invalid_turn'] += 1
            continue
        a, b = a[:2*turn], b[:2*turn]
        if any([x.get('role') for x in c] != ['user','assistant'] * turn or
               any(not isinstance(x.get('content'), str) or not x['content'].strip() for x in c) for c in [a,b]):
            excluded['invalid_conversation'] += 1
            continue
        if any(a[i]['content'] != b[i]['content'] for i in range(0, 2*turn, 2)):
            excluded['question_alignment'] += 1
            continue
        labels = {'model_a':'A', 'model_b':'B', 'tie':'TIE', 'tie (bothbad)':'TIE'}
        if r.get('winner') not in labels or r.get('question_id') is None:
            excluded['invalid_label_or_id'] += 1
            continue
        key = digest([r['question_id'], turn, a, b])
        kept.append({'pair_id':key, 'annotation_id':digest([key, r.get('judge'), idx]),
                     'prompt_id':digest(['mtbench',r['question_id']]), 'source':'mtbench_human',
                     'question_id':r['question_id'], 'turn':turn, 'judge':r.get('judge'),
                     'question':a[-2]['content'], 'conversation_a':a, 'conversation_b':b,
                     'answer_a':a[-1]['content'], 'answer_b':b[-1]['content'],
                     'human_preference':labels[r['winner']]})
    return kept, {'retained_annotations':len(kept), 'unique_pairs':len({r['pair_id'] for r in kept}), 'excluded':dict(excluded)}
