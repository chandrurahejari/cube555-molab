"""Compare an independent CPU replay of three unsaturated TPU beam layers with Molab telemetry."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import zipfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from molab.tpu_symmetry import frame_state, validate_symmetry_tables


def opening_scores(artifacts, weights):
    spec = importlib.util.spec_from_file_location("reference_jax_model", artifacts / "jax_model.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    layout = artifacts / "piece_layout_555.json"
    models = [reference.load_params_from_pt(artifacts / name, layout)
              for name in ("q555_f1_bell2k.pt", "q555_2k_BEST.pt")]
    blend = reference.make_blend(models, weights)

    @jax.jit
    def score(batch):
        return reference.apply(blend, batch, dtype=jnp.bfloat16).astype(jnp.float32)

    def batch_scores(states):
        scores = []
        for start in range(0, len(states), 16):
            batch = states[start:start+16]
            padded = np.pad(batch, ((0, 16-len(batch)), (0, 0)), mode="edge")
            scores.append(np.asarray(score(padded))[:len(batch)])
        return np.concatenate(scores).ravel()

    return batch_scores


def replay(root, generators, score, width):
    names = list(generators)
    inverse = np.array([names.index(n[1:] if n.startswith("-") else "-"+n) for n in names])
    moves = np.array(list(generators.values()))
    owner_weights = np.random.default_rng(12345).integers(0, 2**32-1, 150, dtype=np.uint32) & 7
    hash_weights = np.random.default_rng(0).integers(0, 10**15, 150, dtype=np.int64)
    states = root[None]
    incoming = np.array([-1])
    for depth in range(3):
        scores = score(states)
        children = states[:, moves].reshape(-1, 150)
        hashes = children @ hash_weights
        dest = children @ owner_weights & 7
        source = states @ owner_weights & 7 if depth >= 2 else np.zeros(len(states), dtype=int)
        parent, move = np.divmod(np.arange(len(children)), len(moves))
        valid = np.arange(len(children)) if depth < 2 else np.flatnonzero(inverse[incoming[parent]] != move)

        if depth == 0:
            chosen = np.argsort(scores, kind="stable")
        else:
            route = source[parent[valid]] * 8 + dest[valid]
            if np.bincount(route.astype(int)).max() > 2 * (width // 8) // 8:
                raise ValueError(f"Sender quota is saturated at layer {depth}; this check requires unsaturated layers.")
            # Stable receive deduplication chooses the earliest sender, then its lowest Q.
            ordered = valid[np.lexsort((valid, scores[valid], source[parent[valid]], hashes[valid]))]
            unique = ordered[np.r_[True, hashes[ordered][1:] != hashes[ordered][:-1]]]
            if np.bincount(dest[unique].astype(int)).max() > width // 8:
                raise ValueError(f"Receiver quota is saturated at layer {depth}.")
            chosen = unique[np.lexsort((hashes[unique], scores[unique], dest[unique]))]

        # No seed history is published. Layer two has opposite permutation parity from
        # the first recorded history, so its sole filter before dedup is immediate inverse.
        q = np.sort(np.clip(scores[chosen], 0, 300))
        low, median, high = np.rint(1024 * q[[0, len(q)//2, -1]]).astype(int).tolist()
        yield dict(frontier_size=len(states), generated=len(children), sent=len(valid), unique=len(chosen),
                   kept=len(chosen), q_min_key=low, q_median_key=median, q_cutoff_key=high)
        states, incoming = children[chosen], move[chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--competition", type=Path, required=True, help="Official competition ZIP")
    parser.add_argument("--drive-report", type=Path, required=True, help="Downloaded Molab report with the first three finished layers")
    parser.add_argument("--cube", type=int, help="Overrides puzzle_id when the notebook ran a queue")
    args = parser.parse_args()
    report = json.loads(args.drive_report.read_text(encoding="utf-8"))
    config = report["configuration"]
    cube = config["puzzle_id"] if args.cube is None else args.cube
    if config["score_precision"] != "tpu" or report["tpu_logical_ranks"] != 8:
        parser.error("Expected TPU scoring and eight logical ranks.")
    for name in ("q555_f1_bell2k.pt", "q555_2k_BEST.pt", "piece_layout_555.json"):
        digest = hashlib.sha256((args.artifacts / name).read_bytes()).hexdigest()
        if digest != report["inputs"]["model/"+name]["sha256"]:
            parser.error(f"{name} differs from the file used by the cloud notebook.")
    with zipfile.ZipFile(args.competition) as archive:
        info = json.loads(archive.read("puzzle_info.json"))
        with archive.open("test.csv") as data:
            rows = csv.DictReader(io.TextIOWrapper(data, encoding="utf-8-sig"))
            start = next(row["initial_state"] for row in rows if int(row["initial_state_id"]) == cube)
    tables = json.loads((ROOT / "molab/symmetry-tables.json").read_text())
    validate_symmetry_tables(tables, info)
    start = frame_state(list(map(int, start.split(","))), config["symmetry_index"], config["inverse"], tables)
    observed = {e["depth"]: e for e in report["layer_events"] if e["puzzle_id"] == cube and e["rank"] == 0 and "selection" in e}
    if set(range(3)) - observed.keys():
        parser.error("Missing completed layers 0, 1 or 2 for the requested cube.")
    reference = replay(np.array(start, dtype=np.uint8), info["generators"],
                       opening_scores(args.artifacts, config["blend"]), config["requested_beam"])
    matched = True
    for depth, expected in enumerate(reference):
        cloud = {**observed[depth]["selection"], "frontier_size": observed[depth]["frontier_size"]}
        differences = {k: cloud[k] for k, v in expected.items() if cloud[k] != v}
        print(json.dumps(dict(cube=cube, depth=depth, reference=expected, cloud_differences=differences)), flush=True)
        matched = matched and not differences
    if not matched:
        raise SystemExit("Opening layer telemetry differs from the JAX reference.")


if __name__ == "__main__":
    main()
