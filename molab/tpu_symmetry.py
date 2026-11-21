"""Exact 48-frame transforms from the cited Kaggle TPU notebook and artifact dataset."""
import base64
import csv
import json
from pathlib import Path
import zlib


def symmetry_tables():
    return json.loads(zlib.decompress(base64.b64decode(__SYM_BLOB__)))


def inverse_permutation(values):
    result = [0] * len(values)
    for position, value in enumerate(values):
        result[value] = position
    return result


def frame_state(state, index, inverted, tables):
    transformed = inverse_permutation(state) if inverted else list(state)
    if index:
        symmetry, inverse = tables["sym"][index], tables["sym_inv"][index]
        transformed = [inverse[transformed[position]] for position in symmetry]
    return transformed


def original_path(path, index, inverted, tables):
    names = tables["move_names"]
    tokens = path.split(".") if path else []
    mapping = dict(zip(names, (names[j] for j in tables["relabel_inv"][index])))
    result = [mapping[move] for move in tokens]
    if inverted:
        result = [move[1:] if move.startswith("-") else "-" + move for move in reversed(result)]
    return ".".join(result)


def frame_path(path, index, inverted, tables):
    names = tables["move_names"]
    tokens = path.split(".") if path else []
    if inverted:
        tokens = [move[1:] if move.startswith("-") else "-" + move for move in reversed(tokens)]
    mapping = {names[j]: names[i] for i, j in enumerate(tables["relabel_inv"][index])}
    return ".".join(mapping[move] for move in tokens)


def validate_symmetry_tables(tables, info):
    target = list(range(150))
    if info["central_state"] != target or list(info["generators"]) != tables["move_names"]:
        raise ValueError("Symmetry artifact uses a different puzzle or move ordering")
    names = tables["move_names"]
    if any(len(tables[key]) != 48 for key in ("sym", "sym_inv", "relabel_inv")):
        raise ValueError("Expected the exact 48 TPU symmetry rows")
    for index in range(48):
        symmetry, inverse = tables["sym"][index], tables["sym_inv"][index]
        relabel = tables["relabel_inv"][index]
        if (sorted(symmetry) != target or inverse != inverse_permutation(symmetry)
                or sorted(relabel) != list(range(30))):
            raise ValueError("Invalid symmetry artifact row " + str(index))
        for frame_move, original_move in enumerate(relabel):
            moved = frame_state(info["generators"][names[original_move]], index, False, tables)
            if moved != info["generators"][names[frame_move]]:
                raise ValueError("Symmetry move conjugation mismatch")
    if tables["sym"][0] != target or tables["relabel_inv"][0] != list(range(30)):
        raise ValueError("TPU symmetry zero must be identity")


def prepare_frame_competition(original, destination, index, inverted, tables):
    original, destination = Path(original), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("test.csv", "sample_submission.csv"):
        with (original / filename).open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            with (destination / filename).open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    if filename == "test.csv":
                        state = list(map(int, row["initial_state"].split(",")))
                        row["initial_state"] = ",".join(map(str, frame_state(state, index, inverted, tables)))
                    else:
                        row["path"] = frame_path(row["path"], index, inverted, tables)
                    writer.writerow(row)
    return destination
