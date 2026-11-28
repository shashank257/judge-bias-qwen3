"""Full UltraFeedback study. One active Colab writer per model/mode/chunk."""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .core import digest, read_jsonl, write_jsonl
from .study import semifinals_manifest, finals_manifest
from .uf_trial import LocalJudge, _snapshot, analyze_trial, validate_prompts

CHUNK = {'0.6B': 125, '1.7B': 50, '4B': 25, '8B': 25}


def atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix='.tmp') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


class FullRun:
    def __init__(self, project, drive):
        self.project = Path(project)
        self.out = Path(drive)
        self.prompts = read_jsonl(self.project/'full.jsonl')
        validate_prompts(self.prompts)
        self.configs = json.loads((self.project/'configs.json').read_text())
        self.pilot = read_jsonl(self.project/'pilot.jsonl')
        base = [c for c in self.configs if c['checkpoint'] != 'Qwen/Qwen3-8B']
        contract = digest([self.prompts, base, self.pilot, {k:v for k,v in CHUNK.items() if k!='8B'}, 'full-v1'])
        path = self.out/'contract.json'
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError('Checkpoint folder belongs to a different study.')
        atomic(path, json.dumps(contract).encode())
        extension = digest([contract, self.configs, CHUNK])
        extension_path = self.out/'contract_eight_modes.json'
        if extension_path.exists() and json.loads(extension_path.read_text()) != extension:
            raise ValueError('8B settings differ from existing checkpoints.')
        atomic(extension_path, json.dumps(extension).encode())
        self.model = None
        self.active = None

    def parts(self,c):
        size = CHUNK[c['checkpoint'].split('-')[-1]]
        return [self.prompts[i:i+size] for i in range(0,len(self.prompts),size)]

    def load(self,c,number):
        prompts = self.parts(c)[number-1]
        ids = {p['prompt_id'] for p in prompts}
        path = self.out/c['config_id']/f'part-{number:02}'
        events = [e for e in self.pilot if e['config_id']==c['config_id'] and e['prompt_id'] in ids]
        for batch in sorted(path.glob('batch-*.json')):
            raw = batch.read_bytes()
            if hashlib.sha256(raw).hexdigest() != batch.stem.split('-')[-1]:
                raise ValueError(f'Damaged checkpoint: {batch}')
            events.extend(json.loads(raw))
        return prompts,path,events

    def run(self,mode,number,factory=LocalJudge):
        c = next(c for c in self.configs if c['config_id']==mode)
        parts = self.parts(c)
        if not 1 <= number <= len(parts):
            raise ValueError(f'Invalid part {number} for {mode}')
        marker = self.out/mode/f'part-{number:02}'/'finished.json'
        if marker.is_file():
            saved = json.loads(marker.read_text())
            if (saved['config_id'] == mode and saved['planned_brackets'] == len(parts[number-1])
                    and saved['state'] in ('FINISHED', 'FINISHED WITH MISSING BRACKETS')
                    and saved['semifinals_pending'] == saved['finals_pending'] == 0):
                print(f"Already done: {mode} part {number:02} | {saved['valid_brackets']}/{saved['planned_brackets']} complete brackets. Skipping.")
                return
        prompts,path,events = self.load(c,number)
        latest = {e['request_id']:e for e in events}
        def stage(rows):
            for attempt in (0,1):
                todo=[]
                for r in rows:
                    e=latest.get(r['request_id'])
                    if (attempt==0 and e is None) or (attempt==1 and e and e['attempt']==0 and e['status']=='invalid_verdict'):
                        todo.append(r)
                for i in range(0,len(todo),c['batch_size']):
                    batch=todo[i:i+c['batch_size']]
                    if self.active != mode:
                        if self.model is not None:
                            self.model.close()
                        self.model=None
                        self.active=None
                        if factory is LocalJudge and c['checkpoint']=='Qwen/Qwen3-8B':
                            import torch
                            if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 22*2**30:
                                raise RuntimeError('For 8B FP16 select an L4 24 GB or A100 runtime, then Run all with the same Drive folder.')
                        self.model=factory([c])
                        self.active=mode
                    for r in batch:
                        if len(self.model.encode(r,c))+c['answer_budget']+(c['reasoning_budget']+8 if c['thinking'] else 0)>c['max_context']:
                            raise ValueError('Request exceeds frozen context limit: '+r['request_id'])
                    outputs=self.model.generate_batch(batch,c,attempt)
                    if len(outputs)!=len(batch):
                        raise ValueError('Incomplete batch')
                    records=[dict(r,**v,environment=self.model.metadata,origin='full',utc=datetime.now(timezone.utc).isoformat()) for r,v in zip(batch,outputs)]
                    raw=json.dumps(records,ensure_ascii=False,allow_nan=False).encode()
                    filename=f'batch-{len(events):07}-{hashlib.sha256(raw).hexdigest()}.json'
                    atomic(path/filename,raw)
                    events.extend(records)
                    latest.update({e['request_id']:e for e in records})
                    if i==0 or i//c['batch_size']%10==0:
                        print(f"{mode} part {number} | {batch[0]['match_name']} | attempt {attempt+1} | {min(i+len(batch),len(todo))}/{len(todo)} saved",flush=True)
            return [latest[r['request_id']] for r in rows]
        semis=stage(semifinals_manifest(prompts,[c]))
        finals,_=finals_manifest(prompts,[c],semis)
        stage(finals)
        _,stats=_snapshot(prompts,[c],events)
        atomic(path/'finished.json',json.dumps(stats[0]).encode())
        print(stats[0]['state'],stats[0]['valid_brackets'],'complete brackets out of',len(prompts))

    def export(self):
        selections=[]
        for c in self.configs:
            for number in range(1,len(self.parts(c))+1):
                prompts,path,events=self.load(c,number)
                joined,status=_snapshot(prompts,[c],events)
                if status[0]['semifinals_pending'] or status[0]['finals_pending']:
                    raise ValueError(f"Finish {c['config_id']} part {number} before final export.")
                selections.extend(joined)
        report=self.out/'analysis'
        report.mkdir(exist_ok=True)
        write_jsonl(report/'joined_results.jsonl',selections)
        analyze_trial(self.prompts,self.configs,selections,report)
        (report/'MIDTERM_SUMMARY.md').unlink()
        (report/'run_stats.csv').unlink()  # latest decisions omit earlier attempts
        (report/'ultrafeedback_midterm.png').rename(report/'selection_curves.png')
        import runpy
        runpy.run_path(str(self.project/'pooled.py'),init_globals={'analysis_dir':report})
        print('Saved full-study tables, paired and pooled estimates, figure and judgments to', report)
