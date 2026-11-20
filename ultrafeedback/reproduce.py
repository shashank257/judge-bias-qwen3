from pathlib import Path
import json
import subprocess
import sys
from src.core import read_jsonl
from src.uf_trial import analyze_trial

root = Path(__file__).resolve().parent
data = root / 'data'
configs = json.loads((data / 'suite_metadata.json').read_text())['configs']
analyze_trial(read_jsonl(data / 'selected_prompts.jsonl'), configs,
              read_jsonl(data / 'decisions.jsonl'), root / 'reproduced')
subprocess.run([sys.executable, str(root / 'pooled.py')], check=True)
