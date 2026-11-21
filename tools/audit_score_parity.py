"""Compare the native Cube555 TorchScript heads with the published Kaggle JAX heads."""
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from molab.tpu_symmetry import frame_state, validate_symmetry_tables
from tools.cube555.models import load_blend

FRAMES = ((0, False), (3, False), (23, True), (45, True))
HEADS = ("transformer", "mlp", "blend")


def read_states(archive, submission, cube_ids):
    with zipfile.ZipFile(archive) as data:
        info = json.loads(data.read("puzzle_info.json"))
        with data.open("test.csv") as test:
            reader = csv.DictReader(io.TextIOWrapper(test, encoding="utf-8-sig"))
            starts = {int(r["initial_state_id"]): r["initial_state"] for r in reader}
    with submission.open(encoding="utf-8-sig", newline="") as file:
        paths = {int(r["initial_state_id"]): r["path"] for r in csv.DictReader(file)}
    tables = json.loads((ROOT / "molab/symmetry-tables.json").read_text())
    validate_symmetry_tables(tables, info)
    samples, labels = [], []
    for cube in cube_ids:
        state = np.array(starts[cube].split(","), dtype=np.uint8)
        moves = paths[cube].split(".") if paths[cube] else []
        length = len(moves)
        positions = {0, length // 3, 2 * length // 3, max(0, length - 6)}
        for position in range(length + 1):
            if position in positions:
                for index, reverse in FRAMES:
                    samples.append(frame_state(state.tolist(), index, reverse, tables))
                    labels.append(dict(id=cube, remaining=length-position, symmetry=index, reverse=reverse))
            if position < length:
                state = state[info["generators"][moves[position]]]
        if not np.array_equal(state, info["central_state"]):
            raise ValueError(f"Submission path for cube {cube} does not solve the official state.")
    return np.asarray(samples, dtype=np.uint8), labels


def jax_scores(reference, files, layout, batches, dtype):
    params = [reference.load_params_from_pt(path, layout) for path in files]
    blend = reference.make_blend(params, [0.8, 0.2])

    @jax.jit
    def score(states):
        heads = [reference.apply(p, states, dtype=dtype) for p in [*params, blend]]
        return jnp.stack(heads).astype(jnp.float32)

    return np.concatenate([np.asarray(score(batch)) for batch in batches], axis=1)


def torch_scores(files, layout, batches, dtype, device, mode):
    model = load_blend(*files, layout, 0.8, mode)[0].eval().to(device, dtype=dtype)
    script = torch.jit.script(model)
    scores = []
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for batch in batches:
            # Match PackedState160, with nonzero padding to detect accidental model input.
            rows = np.pad(batch, ((0, 0), (0, 10)), constant_values=199)
            states = torch.from_numpy(rows).to(device)
            heads = (script.transformer(states), script.mlp(states), script(states))
            scores.append(np.stack([t.float().cpu().numpy() for t in heads]))
    return np.concatenate(scores, axis=1)


def compare(reference, native):
    a = np.argsort(reference, axis=1, kind="stable")[:, :3]
    b = np.argsort(native, axis=1, kind="stable")[:, :3]
    return {
        "max_abs": float(np.abs(reference - native).max()),
        "mean_abs": float(np.abs(reference - native).mean()),
        "top1_matches": int(np.count_nonzero(a[:, 0] == b[:, 0])),
        "top3_overlap": sum(len(set(x) & set(y)) for x, y in zip(a, b)),
        "matches_best_among_ties": int(np.sum(reference[np.arange(len(a)), b[:, 0]] == reference.min(axis=1))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True, help="Directory containing jax_model.py, layout and two checkpoint files")
    parser.add_argument("--competition", type=Path, required=True, help="Competition ZIP with puzzle_info.json and test.csv")
    parser.add_argument("--submission", type=Path, required=True, help="CSV with replayable paths")
    parser.add_argument("--cubes", nargs="+", type=int, default=[57, 68, 142, 166, 170, 256, 382, 994])
    parser.add_argument("--torch-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="Save full report, including the list of samples")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.torch_device == "cuda":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    paths = [args.artifacts / name for name in ("q555_f1_bell2k.pt", "q555_2k_BEST.pt")]
    layout = args.artifacts / "piece_layout_555.json"
    jax_path = args.artifacts / "jax_model.py"
    spec = importlib.util.spec_from_file_location("reference_jax_model", jax_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    states, samples = read_states(args.competition, args.submission, args.cubes)
    batches = [states[i:i+8] for i in range(0, len(states), 8)]
    result = {
        "states": len(states), "actions": 30 * len(states), "cubes": args.cubes,
        "torch_device": args.torch_device, "jax_device": str(jax.devices()[0]),
        "versions": {"torch": torch.__version__, "jax": jax.__version__, "numpy": np.__version__},
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths, layout, jax_path, args.submission]},
    }
    fp32 = jax_scores(reference, paths, layout, batches, jnp.float32)
    bf16 = jax_scores(reference, paths, layout, batches, jnp.bfloat16)
    for name, dtype, expected in (("fp32", torch.float32, fp32), ("fp16", torch.float16, bf16),
                                   ("bf16", torch.bfloat16, bf16), ("tpu", torch.bfloat16, bf16)):
        scores = torch_scores(paths, layout, batches, dtype, args.torch_device, "tpu" if name == "tpu" else "fp16")
        ref_name = "fp32" if name == "fp32" else "bf16"
        result[f"jax_{ref_name}__torch_{name}"] = {
            head: compare(a, b) for head, a, b in zip(HEADS, expected, scores)
        }
        if name in ("fp16", "tpu"):
            native = scores[-1]
            if name == "tpu" and not np.array_equal(native, torch.from_numpy(native).bfloat16().float().numpy()):
                raise AssertionError("The native TPU-mode blend must return BF16-representable scores.")
            keys = np.rint(np.clip(native, 0, 300) * 1024) / 1024
            result[name + "_quantization"] = {
                "minimum": float(native.min()), "maximum": float(native.max()),
                "low_clips": int(np.sum(native < 0)), "high_clips": int(np.sum(native > 300)),
                "nonfinite": int(np.sum(~np.isfinite(native))), **compare(native, keys),
            }
            ref_order = np.argsort(bf16[-1].ravel(), kind="stable")
            packed_order = np.argsort(keys.ravel(), kind="stable")
            result[name + "_cross_parent_selection"] = {}
            for k in (len(states), 3 * len(states)):
                result[name + "_cross_parent_selection"][k] = len(set(ref_order[:k]) & set(packed_order[:k]))
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps({**result, "samples": samples}, indent=2), encoding="utf-8")
    if any(values["max_abs"] > 2e-4 for values in result["jax_fp32__torch_fp32"].values()):
        raise SystemExit("FP32 per-head or blend parity check failed.")
    if result["fp16_quantization"]["nonfinite"] or result["tpu_quantization"]["nonfinite"]:
        raise SystemExit("Nonfinite native model output.")


if __name__ == "__main__":
    main()
