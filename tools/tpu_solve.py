"""Run one framed competition cube with the published TPU JAX model and beam search."""
import argparse
import csv
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PINNED = {
    "q555_f1_bell2k.pt": "0241457776212801bc0705b61ffd31a3f97bd8dd44eb9c12ebeee47f51005db2",
    "q555_2k_BEST.pt": "2b540c3e396f7fb5710ccc44201a698740df1761495ee4059be706374e8e5ac2",
    "piece_layout_555.json": "34926e43fea99e6d82cef0bdf1af2ba7cd243daed8ef3409f1cf0b04e457a6d3",
    "bfs_endgame.npz": "2417c256054b14cc5059bf53119484f18524bd3b4448ae4ca5ca99946d08c62d",
}
BFS_URL = ("https://www.kaggle.com/api/v1/datasets/download/artgor/"
           "cube555-tpu-artifacts/bfs_endgame.npz?datasetVersionNumber=3")


def verify(path, digest):
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"{path.name} SHA256 mismatch: expected {digest}, got {actual}")


def endgame_file(assets, output):
    name = "bfs_endgame.npz"
    path = assets / name
    if path.exists():
        verify(path, PINNED[name])
        return path
    # A competition dataset mount can be read-only.
    try:
        temp = tempfile.NamedTemporaryFile(dir=assets, suffix=".download", delete=False)
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise
        path = output / name
        temp = tempfile.NamedTemporaryFile(dir=output, suffix=".download", delete=False)
    staged = Path(temp.name)
    try:
        with temp, urllib.request.urlopen(BFS_URL, timeout=120) as source:
            shutil.copyfileobj(source, temp)
        verify(staged, PINNED[name])
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)
    return path


def read_puzzle(args):
    import numpy as np
    with (args.competition / "test.csv").open(encoding="utf-8-sig", newline="") as file:
        root = next((row["initial_state"] for row in csv.DictReader(file)
                     if int(row["initial_state_id"]) == args.pid), None)
    if root is None:
        raise ValueError(f"Missing cube {args.pid} in test.csv")
    info_path = args.assets / "puzzle_info.json"
    if not info_path.is_file():
        info_path = args.competition / "puzzle_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8-sig"))
    names = list(info["generators"])
    layout = json.loads((args.assets / "piece_layout_555.json").read_text(encoding="utf-8-sig"))
    if names != layout["_move_names"] or info["central_state"] != list(range(150)):
        raise ValueError("Reference model move order and competition puzzle contract differ.")
    start = np.asarray(root.split(","), dtype=np.int64)
    generators = np.array(list(info["generators"].values()), dtype=np.int32)
    goal = np.arange(150, dtype=np.uint8)
    if not np.array_equal(np.sort(start), goal) or not np.all(np.sort(generators, axis=1) == goal):
        raise ValueError("Invalid picture cube or generator permutation.")
    return start.astype(np.uint8), goal, generators, names


def exact_tail(reached, goal, moves, hashes, depths, ztab):
    import numpy as np
    def lookup(states):
        keys = np.bitwise_xor.reduce(ztab[np.arange(150), np.atleast_2d(states)], axis=1)
        indices = np.minimum(np.searchsorted(hashes, keys), len(hashes)-1)
        return np.where(hashes[indices] == keys, depths[indices], -1)

    current = reached
    remaining = int(lookup(current)[0])
    if remaining < 0:
        raise ValueError("Beam prefix did not reach the radius-five endgame table.")
    tail = []
    for distance in range(remaining, 0, -1):
        children = current[moves]
        matching = np.flatnonzero(lookup(children) == distance-1)
        if not len(matching):
            raise ValueError("Endgame table candidate had no verified descending move.")
        move = int(matching[0])
        tail.append(move)
        current = children[move]
    if not np.array_equal(current, goal):
        raise ValueError("Endgame suffix failed replay from the reached state.")
    return tail


def divisor(size, maximum):
    return next(n for n in range(min(size, maximum), 0, -1) if size % n == 0)


def solve(args, emit):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from tools.tpu_reference.jax_model import apply, load_params_from_pt, make_blend
    from tools.tpu_reference.jax_beam_spmd_v_only import beam_solve_v_only_spmd_packed, make_mesh

    if args.layer_audit:
        from tools.tpu_reference.layer_audit import record_layer
        from tools.tpu_reference.observed_engine import load_observed_engine
        from tools.tpu_reference.audit_transport import pack_layer
        previous = None
        def observe(layer, states, packed, scores):
            nonlocal previous
            receipt = record_layer(args.output, layer, states, packed, scores, save_arrays=True)
            emit(dict(event="layer_audit", **receipt))
            previous, packet = pack_layer(args.output / f"layer-{layer:04}.npz", previous, moves)
            emit(dict(event="snapshot", layer=layer, **packet))
        reference = args.original_cpu
        if reference:
            verify(reference / "jax_beam_spmd_v_only.py", "cb62b5e8319b44907114ca1feb06147024afd6c0e4bbc3af5c5fff84501f51f8")
            verify(reference / "jax_model.py", "93955759e1c36031ab983a41249d10406d77eae52b9daee058421a2cde0e86d6")
            sys.path.insert(0, str(reference.resolve()))
        engine = load_observed_engine((reference or (ROOT / "tools/tpu_reference")) / "jax_beam_spmd_v_only.py",
                                      "observed_reference" if reference else "tools.tpu_reference.observed_beam", observe)
        beam_solve_v_only_spmd_packed = engine.beam_solve_v_only_spmd_packed

    device = jax.devices(args.device)[0]
    mesh = make_mesh(jax.devices("cpu") if args.physical_cpu else [device])
    if args.physical_cpu and mesh.size != 8:
        raise ValueError("The original distributed CPU reference requires exactly eight XLA devices")
    emit(dict(event="probe", stage="device_ready", platform=device.platform, device_kind=device.device_kind))
    for name in ("q555_f1_bell2k.pt", "q555_2k_BEST.pt", "piece_layout_555.json"):
        verify(args.assets / name, PINNED[name])
    start, goal, moves, names = read_puzzle(args)
    with np.load(endgame_file(args.assets, args.output), allow_pickle=False) as table:
        hashes, depths, ztab = table["hashes"], table["depths"], table["ztab"]
        if int(table["max_depth"].item()) != 5 or hashes.shape != depths.shape or ztab.shape != (150,150):
            raise ValueError("Invalid radius-five endgame table.")
    layout = args.assets / "piece_layout_555.json"
    blend = make_blend([load_params_from_pt(args.assets / name, layout)
                        for name in ("q555_f1_bell2k.pt", "q555_2k_BEST.pt")], [0.8, 0.2])
    root_q = np.asarray(jax.jit(lambda x: apply(blend, x, dtype=jnp.bfloat16))(start[None])).astype(np.float32)
    if root_q.shape != (1, 30) or not np.isfinite(root_q).all():
        raise ValueError("Invalid Q scores for the initial cube.")
    local = args.beam // 8
    chunk_size = args.parent_chunk if args.parent_chunk is not None else (0 if args.beam <= 4_194_304 else 131072)
    chunk = None if args.original_cpu or chunk_size == 0 else divisor(local, chunk_size)
    micro = divisor(chunk or local, args.model_micro)
    inverse = np.asarray([names.index(n[1:] if n.startswith("-") else "-"+n) for n in names], dtype=np.int32)
    emit(dict(event="probe", stage="model_ready", root_q=root_q[0].tolist(), endgame_count=len(hashes),
              endgame_sha256=PINNED["bfs_endgame.npz"], logical_ranks=8, local_beam=local,
              parent_chunk=chunk, model_micro=micro, k_per_peer=2*local//8))
    hash_vector = np.random.default_rng(0).integers(0, 10**15, 150, dtype=np.int64)
    owner_vector = np.random.default_rng(12345).integers(0, 2**32-1, 150, dtype=np.uint32)
    placement = {} if args.original_cpu else dict(logical_ranks=None if args.physical_cpu else 8, on_layer=emit)
    result = beam_solve_v_only_spmd_packed(
        start.tolist(), blend, jnp.asarray(moves), jnp.asarray(goal), jnp.asarray(hash_vector), mesh,
        B_local=local, K_per_peer=2*local//8, n_gen=30, state_size=150, num_steps=args.depth,
        dtype=jnp.bfloat16, internal_bs=micro, tree_path=str(args.output / "tree.u32"),
        parent_chunk=chunk, owner_hash_vec=jnp.asarray(owner_vector), history_depth=4, history_exact=True,
        eg_hashes=jnp.asarray(hashes), eg_ztab=jnp.asarray(ztab), inv_move_tbl=jnp.asarray(inverse),
        pack_v_score=True, q_mode=True, qv_consistency=0.0, **placement)
    common = dict(compile_seconds=result.get("compile_s"), search_seconds=result.get("wall_s"))
    if not result["found"]:
        return dict(status="not_found", reason="depth_exhausted", **common)
    prefix = [int(i) for i in result["path_idx"]]
    reached = start
    for i in prefix:
        if not 0 <= i < len(moves):
            raise ValueError(f"Invalid move {i} in beam traceback.")
        reached = reached[moves[i]]
    tail = exact_tail(reached, goal, moves, hashes, depths, ztab)
    path = prefix + tail
    check = start
    for i in path:
        check = check[moves[i]]
    if not np.array_equal(check, goal):
        raise ValueError("Combined path failed replay from the framed competition state.")
    def move_names(seq):
        return ".".join(names[i] for i in seq)
    return dict(status="solved", verified=True, path=move_names(path), length=len(path),
                prefix_path=move_names(prefix), prefix_length=len(prefix), tail_path=move_names(tail),
                tail_length=len(tail), **common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("competition", "assets", "output"):
        parser.add_argument("--"+flag, type=Path, required=True)
    for flag in ("pid", "beam", "depth"):
        parser.add_argument("--"+flag, type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--model-micro", type=int, default=4096)
    parser.add_argument("--parent-chunk", type=int, help="Default matches Kaggle: whole rank through 4,194,304, then chunks of 131,072 parents.")
    parser.add_argument("--layer-audit", action="store_true", help="Record selected states, scores and parents for paired TPU/GPU comparisons.")
    parser.add_argument("--original-cpu", type=Path, help="Emulate the unmodified, checksum-pinned Kaggle solver from this artifact directory on eight XLA CPU devices.")
    parser.add_argument("--physical-cpu", action="store_true", help="Exercise the patched physical shard_map on eight local XLA CPU devices.")
    args = parser.parse_args()
    if args.original_cpu:
        if not args.layer_audit:
            parser.error("--original-cpu requires --device cpu --layer-audit")
        args.physical_cpu = True
    if args.physical_cpu:
        if args.device != "cpu":
            parser.error("The CPU placement test requires --device cpu")
        os.environ["JAX_NUM_CPU_DEVICES"] = "8"
    if args.beam < 32 or args.beam % 8 or min(args.depth, args.model_micro) <= 0 or (args.parent_chunk or 0) < 0:
        parser.error("Use a beam divisible by eight (at least 32), and positive depth and batch sizes.")
    os.environ["JAX_PLATFORMS"] = "cpu" if args.device == "cpu" else "cuda"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        parser.error("Result already exists. Choose a new output directory.")
    start = time.monotonic()
    result = dict(pid=args.pid, beam=args.beam, depth=args.depth, layers_completed=0, verified=False,
                  path=None, length=None, prefix_path=None, prefix_length=None, tail_path=None, tail_length=None)
    with (args.output / "events.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        def emit(event):
            row = dict(pid=args.pid, beam=args.beam, unix=time.time(), **event)
            log.write(json.dumps(row, allow_nan=False)+"\n")
            if row["event"] == "layer_end":
                result["layers_completed"] = max(result["layers_completed"], int(row["layer"]))
            elif args.original_cpu and row["event"] == "layer_audit":
                result["layers_completed"] = max(result["layers_completed"], int(row["layer"]))
        try:
            emit(dict(event="probe", stage="worker_started"))
            result.update(solve(args, emit))
        except Exception as error:
            import traceback
            traceback.print_exc()
            result.update(status="error", error=f"{type(error).__name__}: {error}")
        result["elapsed_seconds"] = time.monotonic() - start
        emit(dict(event="result", status=result["status"], elapsed_seconds=result["elapsed_seconds"]))
    staging = args.output / "result.json.tmp"
    staging.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    staging.replace(args.output / "result.json")
    print(json.dumps(result, allow_nan=False), flush=True)
    return int(result["status"] == "error")


if __name__ == "__main__":
    sys.exit(main())
