"""Execute notebooks in place (outputs saved), using the local venv kernel."""
import sys, time, nbformat
from nbclient import NotebookClient
for p in sys.argv[1:]:
    t = time.time(); nb = nbformat.read(p, as_version=4)
    NotebookClient(nb, timeout=None, kernel_name="dsc261", resources={"metadata": {"path": "notebooks"}}).execute()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.write(nb, p); print(f"executed {p} in {time.time()-t:.0f}s")
