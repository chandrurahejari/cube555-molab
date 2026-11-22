"""Attach identical host-side layer measurements to the two published JAX execution paths."""
from pathlib import Path
import types


def load_observed_engine(source, module_name, observer):
    """Return the reference module with one extra output (the already selected Q scores).

    Selection, exchange, parents and goal detection retain their original expressions.
    Both cloud workers fail early if an expected instrumentation point changes.
    """
    source = Path(source)
    text = source.read_text(encoding="utf-8-sig")
    logical = "trace_cutoff_d, active_d) = compiled_step(" in text
    def replace(before, after, count=1):
        nonlocal text
        if text.count(before) != count:
            raise ValueError("Unexpected reference solver source at: " + before)
        text = text.replace(before, after)
    replace("trace_cutoff[None])", "trace_cutoff[None], top_v_keep[None, :])", 2)
    size = 14 if logical else 13
    replace(f'out_specs=(P("cores"),) * {size},', f'out_specs=(P("cores"),) * {size+1},')
    suffix = ", active_d" if logical else ""
    replace("trace_cutoff_d" + suffix + ") = compiled_step(",
            "trace_cutoff_d, selected_q_d" + suffix + ") = compiled_step(")
    seed = "    min_v_log = jnp.full((world_size, num_steps), 1e6, dtype=jnp.float32)"
    # Insert at the packed solver's seed only; the dataset also exposes an unused older solver.
    mark = "    # Seed-move array (populates memmap[0] -- parent_local=0, parent_rank=0)."
    before, after = text.split(mark)
    if after.count(seed) != 1:
        raise ValueError("Unexpected packed seed initialization")
    text = before + mark + after.replace(seed, """
    _root = np.asarray(init_state, dtype=np.uint8).reshape(1, 1, -1)
    _observe(0, np.broadcast_to(_root, (world_size, 1, state_size)), None, None)
    _seed_q = np.zeros(B_local, dtype=np.float32)
    _seed_q[:k0] = np.asarray(values0)[top_idx0_np]
    _seed_p = move0_full_np.astype(np.uint32) << BPTR_MOVE_SHIFT
    _observe(1, np.broadcast_to(states0_np, (world_size, B_local, state_size)),
             np.broadcast_to(_seed_p, (world_size, B_local)), np.broadcast_to(_seed_q, (world_size, B_local)))
""" + seed)
    packed = "            packed_h = np.asarray(packed_d)"
    replace(packed, packed + "\n            _observe(j+1, states_d, packed_h, selected_q_d)")
    module = types.ModuleType(module_name)
    module.__file__ = str(source)
    module.__package__ = module_name.rpartition(".")[0]
    module._observe = observer
    exec(compile(text, str(source), "exec"), module.__dict__)
    return module
