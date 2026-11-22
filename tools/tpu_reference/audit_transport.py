"""Losslessly transfer small, fully observed frontiers through the existing Drive JSON report."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def predict(previous, packed, moves, active):
    slot = packed & 0xffffff
    rank = (packed >> 24) & 7
    move = packed >> 27
    valid = active & (slot < previous.shape[1]) & (move < len(moves))
    result = np.zeros((*packed.shape, 150), dtype=np.uint8)
    result[valid] = np.take_along_axis(previous[rank[valid], slot[valid]], moves[move[valid]], axis=1)
    return result


def pack_layer(path, previous, moves):
    with np.load(path, allow_pickle=False) as file:
        arrays = {name:file[name] for name in file.files}
    states = arrays.pop("states")
    active = np.any(states, axis=-1)
    if previous is None:
        arrays["states"] = states
    else:
        arrays["active"] = np.packbits(active, bitorder="little")
        # Corrections preserve the bytes actually observed even if a parent pointer is wrong.
        arrays["delta"] = states ^ predict(previous, arrays["packed"], moves, active)
    payload = io.BytesIO()
    np.savez_compressed(payload, **arrays)
    return states, dict(data=base64.b64encode(payload.getvalue()).decode("ascii"),
                        states_sha256=hashlib.sha256(states.tobytes()).hexdigest())


def unpack_layers(report, competition_zip, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(competition_zip) as z:
        moves = np.asarray(list(json.loads(z.read("puzzle_info.json"))["generators"].values()))
    previous = None
    for expected, packet in enumerate(report["audit_snapshots"]):
        if packet["layer"] != expected:
            raise ValueError("Missing or reordered Molab snapshot")
        raw = base64.b64decode(packet["data"], validate=True)
        with np.load(io.BytesIO(raw), allow_pickle=False) as file:
            arrays = {key:file[key] for key in file.files}
        if expected:
            shape = arrays["packed"].shape
            active = np.unpackbits(arrays.pop("active"), bitorder="little", count=int(np.prod(shape))).reshape(shape).astype(bool)
            arrays["states"] = arrays.pop("delta") ^ predict(previous, arrays["packed"], moves, active)
        previous = arrays["states"]
        if hashlib.sha256(previous.tobytes()).hexdigest() != packet["states_sha256"]:
            raise ValueError(f"Cloud state bytes failed verification at layer {expected}")
        np.savez_compressed(output / f"layer-{expected:04}.npz", **arrays)
    return len(report["audit_snapshots"])
