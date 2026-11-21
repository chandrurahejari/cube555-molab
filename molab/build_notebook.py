"""Assemble the self-contained Molab notebook from the readable body and patch."""
from pathlib import Path
import argparse
import textwrap
import csv
import hashlib
import json
import base64
import zlib
import uuid
import os
from source_bundle import git, pack_sources, pack_small_inputs

root = Path(__file__).parent
header = '''# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.25"]
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    return json, mo


@app.cell
def _():
    # Original search settings. Change the ID to solve another official cube.
    PUZZLE_ID = __FIRST_CUBE__
    BEAM_WIDTH = __BEAM_WIDTH__
    MAX_DEPTH = __MAX_DEPTH__
    SYMMETRY = __SYMMETRY__
    INVERSE = __INVERSE__
    NOTEBOOK_KEY = __NOTEBOOK_KEY__
    DRIVE_CONFIG = None
    # Focus on complete solutions for the requested cubes.
    CAPACITY_BEAMS = []
    LONGEST_CUBES = __LONGEST_CUBES__
    BASELINE = __BASELINE__
    return BASELINE, BEAM_WIDTH, CAPACITY_BEAMS, DRIVE_CONFIG, INVERSE, LONGEST_CUBES, MAX_DEPTH, NOTEBOOK_KEY, PUZZLE_ID, SYMMETRY


@app.cell
def _():
'''
tail = '''
    return (run_cube555,)


@app.cell
def _(BASELINE, BEAM_WIDTH, CAPACITY_BEAMS, DRIVE_CONFIG, INVERSE, LONGEST_CUBES, MAX_DEPTH, NOTEBOOK_KEY, PUZZLE_ID, SYMMETRY, json, mo, run_cube555):
    def _display(report, path):
        _json = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
        mo.output.replace(mo.vstack([
            mo.md("# Cube555 native на Molab\\n\\n" +
                  "Статус: **" + str(report["status"]) + "**. " +
                  "Решено: " + str(len(report.get("queue_completed", []))) +
                  "/" + str(len(report["queue_ids"])) + ". " +
                  "Прошло: " + str(report.get("elapsed_seconds", 0)) + " с."),
            mo.ui.table([dict(id=a["puzzle_id"], beam=a["effective_beam"],
                             symmetry=a["symmetry"], baseline=a["baseline_length"],
                             length=a.get("solution", {}).get("length"),
                             seconds=a["wall_seconds"], status=a["status"])
                         for a in report["attempts"]], selection=None),
            mo.plain_text("MOLAB_RESULT " + _json),
            mo.download(_json.encode("utf-8"), filename="cube555_gpu_result.json",
                        mimetype="application/json", label="Скачать результат JSON"),
        ]))

    report, report_path = run_cube555(PUZZLE_ID, BEAM_WIDTH, MAX_DEPTH, CAPACITY_BEAMS, _display,
                                     LONGEST_CUBES, BASELINE, SYMMETRY, INVERSE, NOTEBOOK_KEY, DRIVE_CONFIG)
    return report, report_path


if __name__ == "__main__":
    app.run()
'''
def baseline_reference():
    current = root.parent.parent / "outputs/user-solutions/current-lengths.json"
    if current.is_file():
        data = json.loads(current.read_text(encoding="utf-8"))
        lengths = {int(key): value for key, value in data["lengths"].items()}
        if set(lengths) != set(range(1035)) or any(type(n) is not int or n < 0 for n in lengths.values()) or sum(lengths.values()) != data["score"]:
            raise ValueError("Invalid reference length summary")
        return {**data, "lengths": lengths}
    baseline_file = root.parent.parent / "outputs/user-solutions/dataset/shortest_paths_result_1.csv"
    if not baseline_file.is_file():
        return {}
    with baseline_file.open(encoding="utf-8-sig") as stream:
        lengths = {int(row["initial_state_id"]): len(row["path"].split(".")) if row["path"] else 0
                   for row in csv.DictReader(stream)}
    return {"file": baseline_file.name, "sha256": hashlib.sha256(baseline_file.read_bytes()).hexdigest(),
            "score": sum(lengths.values()), "lengths": lengths}


def embedded_body():
    body = (root / "port-body.py").read_text(encoding="utf-8")
    helpers = (root / "tpu_symmetry.py").read_text(encoding="utf-8-sig")
    raw = (root / "symmetry-tables.json").read_bytes()
    blob = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    helpers = helpers.replace("__SYM_BLOB__", repr(blob))
    body = body.replace("    __SYMMETRY_HELPERS__", textwrap.indent(helpers, "    "))
    body = body.replace("    __DRIVE_HELPERS__", textwrap.indent((root / 'drive_sink.py').read_text(encoding='utf-8'), '    '))
    checkpoint = (root / 'checkpoint_publisher.py').read_text(encoding='utf-8')
    drive_transport = root / 'drive_checkpoint.py'
    if drive_transport.is_file():
        checkpoint += '\n' + drive_transport.read_text(encoding='utf-8')
    else:
        checkpoint += '\nclass DriveCheckpoint:\n    def __init__(self, config):\n        raise RuntimeError("Drive checkpoint transport is not packaged in this build")\n'
    body = body.replace("    __CHECKPOINT_HELPERS__", textwrap.indent(checkpoint, '    '))
    repo = root.parent
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    source_hash, source = pack_sources(repo, commit)
    inputs = Path(os.environ.get("CUBE555_INPUTS_ARCHIVE",
                 repo.parent / "work/cube555-source/cube555-inputs-audit.zip"))
    input_hash, input_data = pack_small_inputs(inputs)
    for key, value in {"__SOURCE_COMMIT__": commit, "__SOURCE_SHA256__": source_hash,
                       "__SOURCE_BLOB__": source, "__INPUT_DATA_SHA256__": input_hash,
                       "__INPUT_DATA_BLOB__": input_data}.items():
        body = body.replace(key, repr(value))
    return body


def generate_notebook(destination, puzzle_id, beam_width, max_depth=300, *, queue_ids=None, symmetry=0, inverse=False):
    """Write one self-contained cloud notebook, embedding comparison lengths only."""
    if type(puzzle_id) is not int or not 0 <= puzzle_id <= 1034:
        raise ValueError("Cube ID must be an integer in 0..1034")
    if type(beam_width) is not int or not 1 <= beam_width < 2**64:
        raise ValueError("Beam must be a positive uint64")
    if type(max_depth) is not int or not 1 <= max_depth < 2**32:
        raise ValueError("Maximum depth must be a positive uint32 integer")
    if type(inverse) is not bool or type(symmetry) is not int or symmetry not in range(48):
        raise ValueError("Symmetry must be an index 0..47 and inverse must be boolean")
    selected = list(dict.fromkeys([puzzle_id, *(queue_ids or [])]))
    if any(type(pid) is not int or not 0 <= pid <= 1034 for pid in selected):
        raise ValueError("Every queued cube ID must be an integer in 0..1034")
    body = embedded_body()
    configured = (header.replace("__FIRST_CUBE__", str(puzzle_id))
                        .replace("__BEAM_WIDTH__", str(beam_width))
                        .replace("__MAX_DEPTH__", str(max_depth))
                        .replace("__SYMMETRY__", repr(symmetry))
                        .replace("__INVERSE__", repr(inverse))
                        .replace("__NOTEBOOK_KEY__", repr(uuid.uuid4().hex))
                        .replace("__LONGEST_CUBES__", repr(selected))
                        .replace("__BASELINE__", repr(baseline_reference())))
    text = configured + textwrap.indent(body, "    ") + tail
    target = Path(destination).resolve()
    if target.suffix != ".py":
        raise ValueError("Notebook destination must have a .py extension")
    compile(text, str(target), "exec")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=int)
    parser.add_argument("--beam", type=int, default=3_500_000)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--symmetry", type=int, choices=range(48), default=0)
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--output", type=Path, default=root.parent.parent / "outputs/cube555_molab.py")
    args = parser.parse_args()
    if args.cube is None:
        lengths = baseline_reference().get("lengths", {})
        selected = sorted((pid for pid in lengths if lengths[pid] >= 111), key=lambda pid: (-lengths[pid], pid))
        if not selected:
            parser.error("Provide --cube or a baseline file containing paths at least 111 moves long")
        target = generate_notebook(args.output, selected[0], args.beam, args.depth or 115, queue_ids=selected, symmetry=args.symmetry, inverse=args.inverse)
    else:
        target = generate_notebook(args.output, args.cube, args.beam, args.depth or 300, symmetry=args.symmetry, inverse=args.inverse)
    print(target, target.stat().st_size)


if __name__ == "__main__":
    main()
