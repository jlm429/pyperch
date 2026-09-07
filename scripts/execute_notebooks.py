"""Execute teaching notebooks in fresh kernels using this Python environment.

Run from a source checkout after installing the notebooks extra. Outputs are
saved only after a notebook succeeds. Each notebook is independent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional JSON execution report")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reports = []
    for path in sorted((root / "examples" / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
                cell.metadata.pop("execution", None)
        client = NotebookClient(
            notebook,
            kernel_name="python3",
            timeout=180,
            allow_errors=False,
            record_timing=False,
            resources={"metadata": {"path": str(path.parent)}},
        )
        manager = client.create_kernel_manager()
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        started = perf_counter()
        client.execute()
        elapsed = perf_counter() - started
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        images = sum(
            "image/png" in output.get("data", {})
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.outputs
        )
        report = {
            "notebook": path.name,
            "seconds": round(elapsed, 2),
            "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
            "png_outputs": images,
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
