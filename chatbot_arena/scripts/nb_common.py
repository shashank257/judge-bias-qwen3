"""Helpers to build the week-1 notebooks with nbformat (keeps them diffable)."""
import nbformat as nbf

SETUP = r'''# --- Setup: runs in Google Colab (Drive = persistent storage) or locally ---
import os, sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    # Put this repository at MyDrive/dsc261 (this part runs from chatbot_arena/; outputs persist there).
    os.environ["DSC261_ROOT"] = "/content/drive/MyDrive/dsc261/chatbot_arena"
    get_ipython().run_line_magic("pip", "install -q datasets huggingface_hub pyarrow transformers openai scikit-learn statsmodels nbformat")
    try:
        from google.colab import userdata
        for k in ("HF_TOKEN", "FEATHERLESS_API_KEY", "OPENROUTER_API_KEY", "TOGETHER_API_KEY"):
            try:
                os.environ.setdefault(k, userdata.get(k) or "")
            except Exception:
                pass
    except Exception:
        pass
else:
    os.environ.setdefault("DSC261_ROOT", os.path.abspath(".."))
ROOT = os.environ["DSC261_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 80)
from dsc261 import common as C
print("project root:", ROOT, "| seed:", C.SEED, "| pipeline:", C.PIPELINE_VERSION)
'''


def build(cells, out_path, title):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"provenance": [], "name": title}
    nb.cells = [nbf.v4.new_markdown_cell(cells[0])] + [nbf.v4.new_code_cell(SETUP)]
    for c in cells[1:]:
        if c.startswith("MD:"):
            nb.cells.append(nbf.v4.new_markdown_cell(c[3:].strip()))
        else:
            nb.cells.append(nbf.v4.new_code_cell(c.strip()))
    nbf.write(nb, out_path)
