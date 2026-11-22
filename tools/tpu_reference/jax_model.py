"""Pure-JAX inference for the cube555 30-wide Q heads: ResMLPQ AND PieceTransformerQ555.

Ported from `tetraminx/kaggle_notebooks/tpu_beam_tetraminx/jax_model.py`. Two model
families, both loaded by `load_params_from_pt`, which dispatches on the checkpoint's own
`model_config["model_class"]` -- so the notebook names a file and never an architecture:

  PieceTransformerQ555  `q555_f1_bell2k.pt`, 10,765,087 params -- the 2026-09 head
  ResMLPQ               `q555_2k_BEST.pt` and siblings, 24,757,807 params -- arm A

THE TRANSFORMER (`cube555/src/cube555/piece_model.py::PieceTransformerQ555`):

    token j = sticker_embedding[sticker_local[x[head_j]] + 24 * orbit(head_j)]
              + piece_position_embedding[j] + piece_type_embedding[type_j]
    98 piece tokens + CLS -> input_norm -> 6 pre-norm encoder blocks
      (fused-QKV self-attention, 6 heads of 64, no mask; FF 384 -> 1536 -> 384, SiLU)
    -> output_norm(CLS) -> q_head Linear(384, 30), v_head Linear(384, 1)

  One sticker per piece -- the one on its HEAD facelet -- identifies the whole cubie on
  a picture cube, so a token is ONE embedding lookup into a 7 x 24 = 168-row table.
  This is NOT the tetraminx input stage (a folded K-slot value table); porting that one
  would build a lossy encoding that still runs. The layout that defines `head_j`,
  `orbit` and `type_j` travels in `piece_layout_555.json`, which must be the 98-piece
  file: a 95-piece layout is the cube444 derivation run unmodified and collides the six
  face centres. The loader refuses anything else.

  PRECISION MIRRORS THE BENCHED MODEL, which was NOT autocast: `30_solve.py --bf16`
  calls `load_model(dtype=bfloat16)`, i.e. `model.to(bfloat16)` -- weights, embeddings
  and the residual stream all live in bf16. So `apply(dtype=bf16)` stores every
  activation in bf16 and computes internally in fp32 exactly where the CUDA kernels
  do: LayerNorm statistics, GEMM accumulation (`preferred_element_type=float32`),
  attention scores + softmax (the flash kernel's), SiLU's opmath. With dtype=float32
  every op is fp32, which is the parity gate against PyTorch.

  MEMORY is the transformer's binding constraint on a 16 GB v5e core, not FLOPs. One
  forward holds ~(N, 99, 1536) bf16 FF activations plus (N, 6, 99, 99) fp32 scores
  (lane-padded to 128) -- roughly 0.8 MB per state -- so the notebook runs INTERNAL_BS
  4096 (~3.3 GB) for this model, not the ResMLP's 16384 (~13 GB before the beam's own
  buffers).

THE ResMLP (`cube555/src/cube555/models.py`, 24,757,807 params):

    Embedding(num_classes=150, embed_dim=24)      # picture cube: classes == state_size
      -> flatten                                  # (B, 150*24 = 3600)
      -> Linear(3600, 1024) -> LayerNorm -> ReLU  # input_stack, ONE level
      -> ResBlock x 10                            # lin1 -> ln1 -> ReLU -> lin2 -> ln2
                                                  #   -> ReLU(skip + h)
      -> q_head Linear(1024, 30)                  # one score per generator
      -> v_head Linear(1024, 1)                   # auxiliary, used by qv_consistency

WHY A Q HEAD IS THE WHOLE POINT ON TPU. A scalar-V beam step costs `B x 30` forwards
(score every child); this head scores all 30 children from ONE forward on the parent,
so a step costs `B`. That 30x is what converts directly into beam width, which is the
lever on this puzzle.

SHAPES COME FROM THE CHECKPOINT, not from the call site. Every cube555 checkpoint
embeds `model_config`, so `load_params_from_pt` reads d_model / num_res_blocks /
encoding from the file rather than taking them as arguments that can silently
disagree with the weights. This is deliberately different from the tetraminx loader
(which takes `hidden_dims=` / `num_res_blocks=`): there, a mismatch raises a KeyError
on a missing layer; here it would be a shape error deep in a matmul.

ENCODING. `num_classes` is 150, so one-hot is a 22,500-dim input (44.1M params,
2x slower for no measured gain -- APPROACH.md s2). The shipped checkpoints are all
`encoding="embedding"`. The one-hot branch is kept only so an old checkpoint loads.

Functions the beam engine and notebook import by name:
    apply(params, x, dtype)       -> (B, 30)
    apply_qv(params, x, dtype)    -> ((B, 30), (B,))
    make_blend(members, weights)
    has_value_head(params)
    num_params(params)
    model_kind(params)            -> "piece_transformer_555" | "resmlp" | "blend"
    load_params_from_pt(path, layout_path=None)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

STATE_SIZE = 150
N_GENERATORS = 30
NUM_CLASSES = 150
PIECE_TRANSFORMER = "piece_transformer_555"
LAYOUT_FILE = "piece_layout_555.json"


def _strip_orig_mod(sd: dict) -> dict:
    """Strip torch.compile's '_orig_mod.' key prefix if present."""
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def load_params_from_pt(pt_path: str | Path,
                        layout_path: str | Path | None = None) -> dict[str, Any]:
    """Load any cube555 Q checkpoint and convert it to a JAX params dict.

    Dispatches on the checkpoint's `model_config["model_class"]`: PieceTransformerQ555
    goes to the transformer loader (which needs the piece layout -- by default the
    `piece_layout_555.json` sitting next to the checkpoint); absent or "ResMLPQ" is the
    original ResMLP. Any other class raises instead of being mis-built as a ResMLP.

    Linear weights are transposed (PyTorch (out, in) -> JAX (in, out)) so that
    `h @ w` is the canonical matmul. Everything is returned as float32; cast at the
    call site via `apply(..., dtype=jnp.bfloat16)`.
    """
    import torch

    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    sd = _strip_orig_mod(ckpt["model"] if "model" in ckpt else
                         ckpt.get("state_dict", ckpt))
    mc = dict(ckpt.get("model_config", {}))
    cls = mc.pop("model_class", None)
    if cls == "PieceTransformerQ555":
        if layout_path is None:
            layout_path = Path(pt_path).parent / LAYOUT_FILE
        return _piece_transformer_params(sd, mc, layout_path)
    if cls not in (None, "ResMLPQ"):
        raise ValueError(
            f"{Path(pt_path).name}: model_class {cls!r} has no JAX port here "
            f"(supported: ResMLPQ, PieceTransformerQ555)")

    encoding = mc.get("encoding", "embedding")
    d_model = int(mc.get("d_model", 1024))
    n_rb = int(mc.get("num_res_blocks", 10))
    out_dim = int(mc.get("output_dim", N_GENERATORS))
    state_size = int(mc.get("state_size", STATE_SIZE))
    num_classes = int(mc.get("num_classes", NUM_CLASSES))

    def t(name: str) -> jnp.ndarray:
        return jnp.asarray(sd[name].float().numpy(), dtype=jnp.float32)

    params: dict[str, Any] = {
        "encoding": encoding,
        "state_size": state_size,
        "num_classes": num_classes,
        "d_model": d_model,
        "output_dim": out_dim,
        "input_stack": [],
        "res_blocks": [],
        # `head_*` keeps the tetraminx engine's naming so `q_mode` detection
        # (`int(head_w.shape[-1]) == n_gen`) works unchanged.
        "head_w": jnp.transpose(t("q_head.weight"), (1, 0)),
        "head_b": t("q_head.bias"),
    }
    if encoding == "embedding":
        params["embed"] = t("embedding.weight")          # (num_classes, embed_dim)

    # input_stack is nn.Sequential(Linear, LayerNorm, ReLU): one Linear at index 0,
    # one LayerNorm at index 1. The tetraminx model had one such triple per
    # hidden_dim; ResMLPQ has exactly one, driven by d_model.
    params["input_stack"].append({
        "lin_w": jnp.transpose(t("input_stack.0.weight"), (1, 0)),
        "lin_b": t("input_stack.0.bias"),
        "ln_gamma": t("input_stack.1.weight"),
        "ln_beta": t("input_stack.1.bias"),
    })
    for rb in range(n_rb):
        params["res_blocks"].append({
            "lin1_w": jnp.transpose(t(f"res_blocks.{rb}.lin1.weight"), (1, 0)),
            "lin1_b": t(f"res_blocks.{rb}.lin1.bias"),
            "ln1_gamma": t(f"res_blocks.{rb}.ln1.weight"),
            "ln1_beta": t(f"res_blocks.{rb}.ln1.bias"),
            "lin2_w": jnp.transpose(t(f"res_blocks.{rb}.lin2.weight"), (1, 0)),
            "lin2_b": t(f"res_blocks.{rb}.lin2.bias"),
            "ln2_gamma": t(f"res_blocks.{rb}.ln2.weight"),
            "ln2_beta": t(f"res_blocks.{rb}.ln2.bias"),
        })
    if "v_head.weight" in sd:
        params["value_w"] = jnp.transpose(t("v_head.weight"), (1, 0))
        params["value_b"] = t("v_head.bias")
    return params


def _piece_transformer_params(sd: dict, mc: dict,
                              layout_path: str | Path) -> dict[str, Any]:
    """PieceTransformerQ555 state dict + its 98-piece layout -> JAX params.

    The layout's index tables and the model's shape ints ride along in the dict. The
    beam engine closes over the params (they are not a jit argument), so non-array
    entries are trace-time constants -- which is what lets `apply` dispatch without
    any change to jax_beam_spmd_v_only.py.
    """
    layout_path = Path(layout_path)
    if not layout_path.exists():
        raise FileNotFoundError(
            f"{layout_path} not found. PieceTransformerQ555 reads its 98 piece tokens "
            f"through this layout; put {LAYOUT_FILE} next to the checkpoint or pass "
            f"layout_path=.")
    lay = json.loads(layout_path.read_text(encoding="utf-8"))
    if lay["num_pieces"] != 98:
        raise ValueError(
            f"{layout_path.name}: {lay['num_pieces']} pieces, expected 98. A 95-piece "
            f"layout collides the six face centres and still runs -- rebuild it with "
            f"51_derive_layout_555.py.")

    def t(name: str) -> jnp.ndarray:
        return jnp.asarray(sd[name].float().numpy(), dtype=jnp.float32)

    def lin(prefix: str) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.transpose(t(prefix + ".weight"), (1, 0)), t(prefix + ".bias")

    d_model = int(mc["d_model"])
    n_head = int(mc["nhead"])
    n_layers = int(mc["num_layers"])
    if str(mc.get("activation", "silu")) != "silu":
        raise ValueError(f"activation {mc.get('activation')!r}: only silu is ported")
    vocab = int(lay["sticker_vocab"])
    n_orbits = len(lay["orbit_sizes"])
    emb = t("sticker_embedding.weight")
    if emb.shape != (n_orbits * vocab, d_model):
        raise ValueError(f"sticker_embedding {emb.shape} does not match the layout's "
                         f"{n_orbits} orbits x {vocab} stickers x d_model {d_model}")

    heads = jnp.asarray(lay["head_facelet"], dtype=jnp.int32)             # (98,)
    orbit = jnp.asarray(lay["sticker_orbit"], dtype=jnp.int32)            # (150,)
    blocks = []
    for i in range(n_layers):
        p = f"blocks.{i}"
        qkv_w, qkv_b = jnp.transpose(t(p + ".attn.in_proj_weight"), (1, 0)), \
            t(p + ".attn.in_proj_bias")
        o_w, o_b = lin(p + ".attn.out_proj")
        ff1_w, ff1_b = lin(p + ".ff.0")
        ff2_w, ff2_b = lin(p + ".ff.2")          # ff.1 is the SiLU: no parameters
        blocks.append({
            "n1_g": t(p + ".norm1.weight"), "n1_b": t(p + ".norm1.bias"),
            "qkv_w": qkv_w, "qkv_b": qkv_b, "o_w": o_w, "o_b": o_b,
            "n2_g": t(p + ".norm2.weight"), "n2_b": t(p + ".norm2.bias"),
            "ff1_w": ff1_w, "ff1_b": ff1_b, "ff2_w": ff2_w, "ff2_b": ff2_b,
        })
    head_w, head_b = lin("q_head")
    params: dict[str, Any] = {
        "kind": PIECE_TRANSFORMER,
        "d_model": d_model, "n_head": n_head,
        "heads": heads,
        "sticker_local": jnp.asarray(lay["sticker_local"], dtype=jnp.int32),  # (150,)
        # orbit(head_j) * 24: the ORBIT offset, for this embedding arm. The one-hot
        # arm's slot offset (j * 24) is a different table; see piece_model.py.
        "orbit_offset": orbit[heads] * vocab,
        "piece_types": jnp.asarray(lay["piece_types"], dtype=jnp.int32),
        "sticker_emb": emb,
        "pos_emb": t("piece_position_embedding.weight"),
        "type_emb": t("piece_type_embedding.weight"),
        "cls": t("cls_token").reshape(1, 1, d_model),
        "in_g": t("input_norm.weight"), "in_b": t("input_norm.bias"),
        "blocks": blocks,
        "out_g": t("output_norm.weight"), "out_b": t("output_norm.bias"),
        # `head_*` keeps the engine's q_mode detection (head width == n_gen) working.
        "head_w": head_w, "head_b": head_b,
    }
    if "v_head.weight" in sd:
        params["value_w"], params["value_b"] = lin("v_head")
    return params


def _layer_norm(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
                eps: float = 1e-5) -> jnp.ndarray:
    """PyTorch nn.LayerNorm semantics over the last dim."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * gamma + beta


def _ln_as(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, dtype,
           eps: float = 1e-5) -> jnp.ndarray:
    """nn.LayerNorm on a `dtype` model: fp32 statistics, params rounded to `dtype`,
    output in `dtype` -- what PyTorch's CUDA kernel does for a bf16 module."""
    f32 = jnp.float32
    x32 = x.astype(f32)
    mean = jnp.mean(x32, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x32 - mean), axis=-1, keepdims=True)
    y = (x32 - mean) * jax.lax.rsqrt(var + eps)
    y = y * gamma.astype(dtype).astype(f32) + beta.astype(dtype).astype(f32)
    return y.astype(dtype)


def _linear(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray, dtype) -> jnp.ndarray:
    """nn.Linear on a `dtype` model: `dtype` GEMM, fp32 accumulate, output in `dtype`."""
    y = jnp.matmul(x.astype(dtype), w.astype(dtype), preferred_element_type=jnp.float32)
    return (y + b.astype(dtype).astype(jnp.float32)).astype(dtype)


def _piece_transformer_forward(params: dict[str, Any], x: jnp.ndarray, dtype,
                               want_value: bool = False):
    """x (B, 150) integer state -> Q (B, 30) [, V (B,)] on a `dtype` model."""
    f32 = jnp.float32
    B = x.shape[0]
    D, nh = params["d_model"], params["n_head"]
    dh = D // nh
    T = params["heads"].shape[0] + 1                         # 98 pieces + CLS

    # (B, 98) sticker ON each piece's head facelet -> row of the 7x24 orbit table.
    s = jnp.take(x.astype(jnp.int32), params["heads"], axis=1)
    tok = jnp.take(params["sticker_local"], s, axis=0) + params["orbit_offset"][None, :]
    h = jnp.take(params["sticker_emb"].astype(dtype), tok, axis=0)   # (B, 98, D)
    h = h + params["pos_emb"].astype(dtype)[None]
    h = h + jnp.take(params["type_emb"].astype(dtype), params["piece_types"], axis=0)[None]
    h = jnp.concatenate(
        [jnp.broadcast_to(params["cls"].astype(dtype), (B, 1, D)), h], axis=1)
    h = _ln_as(h, params["in_g"], params["in_b"], dtype)

    scale = f32(1.0 / (dh ** 0.5))                            # SDPA's default scale
    for blk in params["blocks"]:
        z = _ln_as(h, blk["n1_g"], blk["n1_b"], dtype)
        qkv = _linear(z, blk["qkv_w"], blk["qkv_b"], dtype)  # (B, T, 3D)
        qkv = qkv.reshape(B, T, 3, nh, dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]   # (B, T, nh, dh)
        # The flash kernel's numerics: fp32 scores and softmax, P rounded to the
        # input dtype for P @ V, fp32 accumulation, output in the input dtype.
        sc = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=f32) * scale
        p = jax.nn.softmax(sc, axis=-1)
        o = jnp.einsum("bhqk,bkhd->bqhd", p.astype(dtype), v,
                       preferred_element_type=f32).astype(dtype)
        h = h + _linear(o.reshape(B, T, D), blk["o_w"], blk["o_b"], dtype)
        z = _ln_as(h, blk["n2_g"], blk["n2_b"], dtype)
        f = _linear(z, blk["ff1_w"], blk["ff1_b"], dtype)
        f = jax.nn.silu(f.astype(f32)).astype(dtype)          # opmath fp32, stored dtype
        h = h + _linear(f, blk["ff2_w"], blk["ff2_b"], dtype)

    pooled = _ln_as(h[:, 0], params["out_g"], params["out_b"], dtype)
    q = _linear(pooled, params["head_w"], params["head_b"], dtype)
    if not want_value:
        return q
    if "value_w" not in params:
        raise ValueError("apply_qv needs a checkpoint with v_head.* (az_head=True)")
    v = _linear(pooled, params["value_w"], params["value_b"], dtype)
    return q, jnp.squeeze(v, axis=-1)


def _trunk(params: dict[str, Any], x: jnp.ndarray, dtype) -> jnp.ndarray:
    """Shared trunk -> (B, d_model). Both heads read this, so qv is one pass."""
    if params.get("encoding", "embedding") == "onehot":
        oh = jax.nn.one_hot(x.astype(jnp.int32), params["num_classes"], dtype=dtype)
        h = oh.reshape(oh.shape[0], -1)
    else:
        # x is uint8 (0..149 -- see the state-dtype note in jax_beam_spmd_v_only.py).
        # int8 would wrap classes 128..149 negative and index the embedding from the
        # far end without erroring, which is why the cast target is explicit here.
        e = params["embed"][x.astype(jnp.int32)]          # (B, S, embed_dim) f32
        h = e.reshape(e.shape[0], -1).astype(dtype)

    for layer in params["input_stack"]:
        h = h @ layer["lin_w"].astype(dtype) + layer["lin_b"].astype(dtype)
        h = _layer_norm(h, layer["ln_gamma"].astype(dtype),
                        layer["ln_beta"].astype(dtype))
        h = jax.nn.relu(h)

    for rb in params["res_blocks"]:
        skip = h
        h1 = h @ rb["lin1_w"].astype(dtype) + rb["lin1_b"].astype(dtype)
        h1 = _layer_norm(h1, rb["ln1_gamma"].astype(dtype), rb["ln1_beta"].astype(dtype))
        h1 = jax.nn.relu(h1)
        h2 = h1 @ rb["lin2_w"].astype(dtype) + rb["lin2_b"].astype(dtype)
        h2 = _layer_norm(h2, rb["ln2_gamma"].astype(dtype), rb["ln2_beta"].astype(dtype))
        h = jax.nn.relu(skip + h2)
    return h


def apply(params: dict[str, Any], x: jnp.ndarray,
          dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """Forward pass. x: (B, state_size) integer state. Returns (B, output_dim).

    output_dim is 30 here, so this is NOT squeezed -- the squeeze-to-(B,) branch of
    the tetraminx original only fired for a scalar V head, which this file does not
    ship. The engine's `q_mode` path expects the (B, n_gen) shape.
    """
    if "members" in params:                      # output-space blend
        acc = None
        for p, w in zip(params["members"], params["weights"]):
            o = apply(p, x, dtype=dtype).astype(jnp.float32) * jnp.float32(w)
            acc = o if acc is None else acc + o
        return acc.astype(dtype)
    if params.get("kind") == PIECE_TRANSFORMER:
        return _piece_transformer_forward(params, x, dtype)
    h = _trunk(params, x, dtype)
    out = h @ params["head_w"].astype(dtype) + params["head_b"].astype(dtype)
    if out.shape[-1] == 1:
        out = jnp.squeeze(out, axis=-1)
    return out


def apply_qv(params: dict[str, Any], x: jnp.ndarray,
             dtype: jnp.dtype = jnp.float32) -> tuple[jnp.ndarray, jnp.ndarray]:
    """-> (Q (B, n_gen), V (B,)) from ONE trunk pass.

    That single pass is what makes `qv_consistency` free rather than a second forward.
    For a blend the Q side is the weighted average, but V is taken from the FIRST
    member that has a value head rather than averaged: value heads are trained
    separately and are not calibrated to a common scale.
    """
    if "members" in params:
        q = v = None
        for p, w in zip(params["members"], params["weights"]):
            if v is None and has_value_head(p):
                qi, v = apply_qv(p, x, dtype=dtype)
            else:
                qi = apply(p, x, dtype=dtype)
            qi = qi.astype(jnp.float32) * jnp.float32(w)
            q = qi if q is None else q + qi
        if v is None:
            raise ValueError("no blend member has a value head")
        return q.astype(dtype), v
    if params.get("kind") == PIECE_TRANSFORMER:
        return _piece_transformer_forward(params, x, dtype, want_value=True)
    if "value_w" not in params:
        raise ValueError("apply_qv needs a checkpoint with v_head.* (az_head=True)")
    h = _trunk(params, x, dtype)
    q = h @ params["head_w"].astype(dtype) + params["head_b"].astype(dtype)
    v = h @ params["value_w"].astype(dtype) + params["value_b"].astype(dtype)
    return q, jnp.squeeze(v, axis=-1)


def make_blend(members: list[dict[str, Any]],
               weights: list[float] | None = None) -> dict[str, Any]:
    """Output-space ensemble of Q models, scored jointly at every beam step.

    Legitimate here only because every cube555 checkpoint is the same architecture
    warm-started from the same pretrained parent, so the members share a distance
    scale by construction. Do NOT blend across symmetry frames -- raw scores from
    different frames are not comparable.

    Note RESULTS.md s2 measures the checkpoints as COMPLEMENTARY (min-merge over six
    arms: 24/24 vs 18/24 for the best single one). That is a per-pid min over separate
    runs, which is not the same thing as averaging scores inside one beam step; this
    blend is the untested variant, so it defaults off in the notebook.
    """
    if not members:
        raise ValueError("make_blend needs at least one member")
    w = list(weights) if weights is not None else [1.0] * len(members)
    if len(w) != len(members):
        raise ValueError(f"{len(w)} weights for {len(members)} members")
    tot = float(sum(w))
    if tot <= 0:
        raise ValueError("blend weights must sum to > 0")
    return {"kind": "blend", "members": members, "weights": [x / tot for x in w]}


def has_value_head(params: dict[str, Any]) -> bool:
    """Can apply_qv() serve a parent value for this params tree?"""
    if "members" in params:
        return any(has_value_head(p) for p in params["members"])
    return "value_w" in params


def model_kind(params: dict[str, Any]) -> str:
    """"piece_transformer_555", "resmlp" or "blend" -- for the notebook's sizing rules."""
    if "members" in params:
        return "blend"
    return params.get("kind", "resmlp")


def num_params(params: dict[str, Any]) -> int:
    """Count parameters (ResMLPQ 24,757,807; PieceTransformerQ555 10,765,087)."""
    if "members" in params:
        return sum(num_params(p) for p in params["members"])
    if params.get("kind") == PIECE_TRANSFORMER:
        # Weights only: the layout's int32 index tables are not parameters.
        skip = {"heads", "sticker_local", "orbit_offset", "piece_types"}
        n = sum(int(v.size) for k, v in params.items()
                if k not in skip and hasattr(v, "size") and k != "blocks")
        return n + sum(int(v.size) for blk in params["blocks"] for v in blk.values())
    n = int(params["embed"].size) if "embed" in params else 0
    for layer in params["input_stack"]:
        n += sum(int(v.size) for v in layer.values())
    for rb in params["res_blocks"]:
        n += sum(int(v.size) for v in rb.values())
    n += int(params["head_w"].size) + int(params["head_b"].size)
    if "value_w" in params:
        n += int(params["value_w"].size) + int(params["value_b"].size)
    return n
