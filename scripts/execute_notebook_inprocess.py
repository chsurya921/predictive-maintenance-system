"""
scripts/execute_notebook_inprocess.py
=====================================
Executes a Jupyter notebook in-process without requiring external ZMQ sockets.
Captures stdout and returns fully executed notebook with cell outputs populated.

Author: ML Engineering Team
"""

import sys
import io
import contextlib
import traceback
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_notebook(nb_path: Path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    # Execution namespace
    exec_globals = {
        "__name__": "__main__",
        "__file__": str(nb_path),
    }

    execution_count = 1
    total_cells = len(nb.cells)
    print(f"Executing {total_cells} cells in {nb_path.name}...")

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        cell.outputs = []
        cell.execution_count = execution_count
        execution_count += 1

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(cell.source, exec_globals)
        except Exception:
            err_msg = traceback.format_exc()
            stderr_buf.write(err_msg)
            print(f"Error executing cell {i}:\\n{err_msg}")

        out_text = stdout_buf.getvalue()
        err_text = stderr_buf.getvalue()

        if out_text:
            cell.outputs.append(nbf.v4.new_output(output_type="stream", name="stdout", text=out_text))
        if err_text:
            cell.outputs.append(nbf.v4.new_output(output_type="stream", name="stderr", text=err_text))

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook {nb_path.name} successfully executed and saved with outputs!")

if __name__ == "__main__":
    target = PROJECT_ROOT / "notebooks" / "exploratory_analysis.ipynb"
    run_notebook(target)
