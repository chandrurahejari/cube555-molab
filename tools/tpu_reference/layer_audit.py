"""Write comparable beam observations without retaining more than one device rank on the host."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
from collections import Counter
from contextlib import nullcontext

import numpy as np


def _digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _fingerprints(rows):
    """Two uint64 fingerprints over 19 little-endian words, with the last two bytes zero."""
    padded = np.zeros((len(rows), 152), dtype=np.uint8)
    padded[:, :150] = rows
    words = padded.view("<u8")
    a = np.full(len(rows), 0xcbf29ce484222325, dtype=np.uint64)
    b = np.full(len(rows), 0x9e3779b97f4a7c15, dtype=np.uint64)
    for i in range(19):
        a ^= words[:, i]
        a *= np.uint64(0x100000001b3)
        b ^= words[:, 18-i]
        b *= np.uint64(0xc6a4a7935bd1e995)
        b ^= b >> np.uint64(47)
    return a, b


def _score_stats(values):
    # Convert to BF16 using round-to-nearest, ties-to-even. Canonicalize all NaNs.
    source = np.ascontiguousarray(values, dtype="<f4")
    bits = source.view("<u4")
    bf16 = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype("<u2")
    bf16[np.isnan(source)] = 0x7fc0
    counts = np.bincount(bf16, minlength=65536).astype("<u4")
    populated = np.flatnonzero(counts)
    top = populated[np.lexsort((populated, -counts[populated].astype(np.int64)))[:24]]
    bins = [[f"{i:04x}", int(counts[i])] for i in top]
    rounded = (bf16.astype("<u4") << 16).view("<f4")
    finite = rounded[np.isfinite(rounded)]
    quartiles = None
    if len(finite):
        positions = [0, len(finite)//4, len(finite)//2, 3*len(finite)//4, len(finite)-1]
        quartiles = np.partition(finite, positions)[positions].tolist()
    return bf16, dict(bf16_hist_sha256=_digest(counts), bf16_top_bins=bins, bf16_bins=len(populated),
                      bf16_other_count=len(source)-sum(count for _, count in bins),
                      nonfinite_count=len(source)-len(finite), bf16_five_number=quartiles)


def _rank(rank, states, packed, scores, snapshots):
    rows = np.asarray(states[rank])
    if rows.dtype != np.uint8:
        raise ValueError("Beam states must be uint8.")
    if snapshots:
        snapshots["states"][rank] = rows
    # A valid picture-cube permutation may start with sticker zero. Only an entirely
    # zero row is TPU padding.
    active = np.any(rows, axis=1)
    count = int(np.count_nonzero(active))
    ordered = hashlib.sha256()
    fingerprints = np.empty((count, 2), dtype="<u8")
    offset = 0
    for start in range(0, len(rows), 8192):
        block = rows[start:start+8192][active[start:start+8192]]
        if len(block):
            ordered.update(memoryview(block).cast("B"))
            batch = fingerprints[offset:offset+len(block)]
            batch[:, 0], batch[:, 1] = _fingerprints(block)
            offset += len(block)
    order = np.lexsort((fingerprints[:, 1], fingerprints[:, 0]))
    totals = [int(i) for i in fingerprints.sum(axis=0, dtype=np.uint64)]
    entry = dict(rank=rank, active_count=count, slots=len(rows), states_sha256=ordered.hexdigest(),
                 active_slots_sha256=_digest(np.packbits(active, bitorder="little")),
                 sorted_fingerprints_sha256=_digest(fingerprints[order].reshape(-1)),
                 multiset_sum64=[f"{h:016x}" for h in totals])
    if packed is not None:
        pointers = np.asarray(packed[rank])
        if pointers.dtype.kind != "u" or pointers.dtype.itemsize != 4:
            raise ValueError("Packed backpointers must be uint32.")
        if snapshots:
            snapshots["packed"][rank] = pointers
        entry["packed_sha256"] = _digest(pointers[active].astype("<u4", copy=False))
    bf16 = None
    if scores is not None:
        score_row = np.asarray(scores[rank], dtype="<f4")
        if snapshots:
            snapshots["scores"][rank] = score_row
        bf16, stats = _score_stats(score_row[active])
        entry.update(stats)
        entry["ordered_scores_sha256"] = _digest(bf16)
        entry["scores_by_fingerprint_sha256"] = _digest(bf16[order])
    # Sample by state identity to make the same stickers inspectable after a reorder.
    slots = np.flatnonzero(active)
    entry["samples"] = []
    for pos in order[:4]:
        slot = int(slots[pos])
        sample = dict(slot=slot, sticker_sha256=_digest(rows[slot]))
        if bf16 is not None:
            sample["score_bf16"] = f"{bf16[pos]:04x}"
        entry["samples"].append(sample)
    return entry, totals


def record_layer(directory, layer, states, packed, scores=None, save_arrays=False):
    """Append one JSONL record to directory/layers.jsonl, fsync it, and return it.

    states has shape (8, B, 150). Optional packed and scores have shape (8, B).
    Each run owns its output directory. SHA256 fields use little-endian byte order.
    """
    shape = tuple(states.shape)
    if len(shape) != 3 or shape[0] != 8 or shape[2] != 150 or any(
        data is not None and tuple(data.shape) != shape[:2] for data in (packed, scores)
    ):
        raise ValueError("Expected states (8,B,150) and optional packed/scores (8,B).")
    entries, combined = [], [0, 0]
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    # Write rank-by-rank to temporary NPY memmaps, then stream those into a standard NPZ.
    with tempfile.TemporaryDirectory(dir=path) if save_arrays else nullcontext() as temp:
        snapshots = {}
        if save_arrays:
            for name, data, dtype in (("states", states, "u1"), ("scores", scores, "<f4"), ("packed", packed, "<u4")):
                if data is not None:
                    snapshots[name] = np.lib.format.open_memmap(Path(temp) / (name+".npy"),
                                                                mode="w+", dtype=dtype, shape=data.shape)
        try:
            for rank in range(8):
                entry, totals = _rank(rank, states, packed, scores, snapshots)
                entries.append(entry)
                for i in range(2):
                    combined[i] = (combined[i] + totals[i]) & ((1 << 64)-1)
            if save_arrays:
                for array in snapshots.values():
                    array.flush()
                staged = Path(temp) / "state.npz"
                with staged.open("wb") as fp:
                    with zipfile.ZipFile(fp, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                        for name in snapshots:
                            archive.write(Path(temp) / (name+".npy"), arcname=name+".npy")
                    fp.flush()
                    os.fsync(fp.fileno())
                staged.replace(path / f"layer-{int(layer):04d}.npz")
        finally:
            for array in snapshots.values():
                array._mmap.close()
    result = dict(version=1, layer=int(layer), active_count=sum(r["active_count"] for r in entries),
                  multiset_sum64=[f"{h:016x}" for h in combined], ranks=entries)
    payload = (json.dumps(result, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")
    # One write to a regular O_APPEND file publishes the entire record on Linux.
    fd = os.open(path / "layers.jsonl", os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
        if os.write(fd, payload) != len(payload):
            raise OSError("Short write while appending the layer audit.")
        os.fsync(fd)
    finally:
        os.close(fd)
    return result


def _slot_details(data, rank, slot):
    row = data["states"][rank, slot]
    result = dict(rank=int(rank), slot=int(slot), padding=not bool(row.any()), stickers=row.tolist())
    if "scores" in data:
        score = data["scores"][rank, slot]
        result["q_float32_hex"] = f"{int(score.view('<u4')):08x}"
        result["q"] = float(score) if np.isfinite(score) else str(score)
    if "packed" in data:
        p = int(data["packed"][rank, slot])
        result["backpointer"] = dict(packed=p, parent_slot=p & 0xffffff, parent_rank=(p >> 24) & 7, move=p >> 27)
    return result


def _load_layer(path):
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key] for key in file.files}


def _state_views(states):
    return np.ascontiguousarray(states).view("V150").reshape(states.shape[:2])


def _multiset(states):
    return Counter(row.tobytes() for row in states.ravel())


def _candidate_scores(layer, parent_layer):
    """Identify the scored (parent permutation, move) independent of rank and slot."""
    result = {}
    states, packed, scores = layer["states"], layer["packed"], layer["scores"]
    active = np.any(states, axis=-1) & (scores < 5e8)
    for rank, slot in np.argwhere(active):
        pointer = int(packed[rank, slot])
        owner, index = (pointer >> 24) & 7, pointer & 0xffffff
        parents = parent_layer["states"]
        if owner >= parents.shape[0] or index >= parents.shape[1] or pointer >> 27 >= 30:
            continue
        parent = parents[owner, index]
        key = (parent.tobytes(), pointer >> 27)
        result[key] = float(scores[rank, slot])
    return result


def _compare_layer(left, right):
    a, b = left["states"], right["states"]
    if a.shape != b.shape:
        return dict(left_shape=a.shape, right_shape=b.shape)
    av, bv = _state_views(a), _state_views(b)
    mismatches = av != bv
    result = {}
    first = None
    if np.any(mismatches):
        rank, slot = np.argwhere(mismatches)[0]
        first = (rank, slot)
        am, bm = _multiset(av), _multiset(bv)
        missing, extra = am-bm, bm-am
        result.update(different_slots=int(np.sum(mismatches)), left_only_count=sum(missing.values()),
                      right_only_count=sum(extra.values()), same_global_multiset=not bool(missing or extra),
                      different_rank_multisets=[r for r in range(len(av)) if _multiset(av[r]) != _multiset(bv[r])])
        for key, diff in (("left_only_example", missing), ("right_only_example", extra)):
            if diff:
                row = next(iter(diff))
                result[key] = dict(stickers=list(row), excess_multiplicity=diff[row])
    for key in ("packed", "scores"):
        if (key in left) != (key in right):
            result[key] = "present_only_on_left" if key in left else "present_only_on_right"
        elif key in left:
            x, y = left[key].view("<u4"), right[key].view("<u4")
            if x.shape != y.shape:
                result[key] = dict(left_shape=x.shape, right_shape=y.shape)
                continue
            unequal = x != y
            if np.any(unequal):
                rank, slot = np.argwhere(unequal)[0]
                changed = dict(different_slots=int(np.sum(unequal)),
                               different_same_sticker_row=int(np.sum(unequal & ~mismatches)),
                               first_left=_slot_details(left, rank, slot),
                               first_right=_slot_details(right, rank, slot))
                if key == "scores":
                    finite = np.isfinite(left[key]) & np.isfinite(right[key]) & ~mismatches
                    if np.any(finite):
                        changed["max_abs_on_same_stickers"] = float(np.max(np.abs(
                            left[key][finite].astype(np.float64) - right[key][finite])))
                result[key] = changed
                if first is None:
                    first = (rank,slot)
    if first is not None:
        result["first_left"] = _slot_details(left, *first)
        result["first_right"] = _slot_details(right, *first)
    return result


def compare_directories(left, right):
    """Compare every saved layer and report first differences in scores and frontier separately."""
    def snapshots(path):
        return {int(p.stem.split("-")[-1]): p for p in Path(path).glob("layer-*.npz")}
    left_files, right_files = snapshots(left), snapshots(right)
    if not left_files and not right_files:
        raise ValueError("Both audit directories are missing layer snapshots.")
    checked, layers = [], []
    first = first_frontier = first_scores = None
    first_detail = {}
    prev_a = prev_b = None
    for layer in sorted(left_files.keys() | right_files.keys()):
        if layer not in left_files or layer not in right_files:
            direction = "left" if layer not in left_files else "right"
            layers.append(dict(layer=layer, missing_from=direction))
            if first is None:
                first, first_detail = layer, dict(missing_from=direction)
            prev_a = prev_b = None
            continue
        a, b = _load_layer(left_files[layer]), _load_layer(right_files[layer])
        difference = _compare_layer(a, b)
        summary = dict(layer=layer, same_slots=not ("different_slots" in difference or "left_shape" in difference),
                       active_left=int(np.any(a["states"],axis=-1).sum()), active_right=int(np.any(b["states"],axis=-1).sum()),
                       left_only=difference.get("left_only_count", 0), right_only=difference.get("right_only_count", 0),
                       different_pointer_slots=difference.get("packed", {}), different_q_slots=difference.get("scores", {}))
        if a["states"].shape!=b["states"].shape:
            left_counts = _multiset(_state_views(a["states"]))
            right_counts = _multiset(_state_views(b["states"]))
            summary.update(left_only=sum((left_counts-right_counts).values()), right_only=sum((right_counts-left_counts).values()))
        # Zero rows are padding. Count only legal sticker rows when comparing frontiers.
        a_pad = a["states"].shape[0]*a["states"].shape[1]-summary["active_left"]
        b_pad = b["states"].shape[0]*b["states"].shape[1]-summary["active_right"]
        summary["left_only"] -= max(0,a_pad-b_pad)
        summary["right_only"] -= max(0,b_pad-a_pad)
        if first_frontier is None and (summary["left_only"] or summary["right_only"]):
            first_frontier = layer
        if first_scores is None and difference.get("scores"):
            first_scores = layer
        if first is None and difference:
            first, first_detail = layer, dict(difference=difference)
        if not difference:
            checked.append(layer)
        if prev_a is not None and all("packed" in view and "scores" in view for view in (a, b)):
            scores_a, scores_b = _candidate_scores(a, prev_a), _candidate_scores(b, prev_b)
            shared = scores_a.keys() & scores_b.keys()
            unequal = sorted(k for k in shared if scores_a[k] != scores_b[k])
            summary.update(shared_parent_moves=len(shared), same_parent_move_q_differences=len(unequal))
            if unequal:
                key = unequal[0]
                summary["q_example"] = dict(parent_stickers=list(key[0]), move_index=key[1],
                                            left=scores_a[key], right=scores_b[key])
        layers.append(summary)
        prev_a, prev_b = a, b
    return dict(equal=first is None, checked=checked, first_unequal=first, first_frontier_difference=first_frontier,
                first_score_slot_difference=first_scores, layers=layers, **first_detail)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare all captured slots from two reference beam runs.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    comparison = compare_directories(args.left, args.right)
    print(json.dumps(comparison, indent=2, allow_nan=False))
    raise SystemExit(int(not comparison["equal"]))
