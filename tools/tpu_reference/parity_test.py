"""Compare the original eight-device JAX step to eight logical ranks on one device.

Run as python -m tools.tpu_reference.parity_test. It uses CPU devices and
small synthetic permutations to exercise every packed step, including ties.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_NUM_CPU_DEVICES"] = "8"
import tempfile
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from . import jax_beam_spmd_v_only as beam


def main():
    physical = jax.devices("cpu")
    if len(physical) != 8:
        raise RuntimeError(f"Eight CPU devices required, got {len(physical)}")
    rng = np.random.default_rng(123)
    solved = np.arange(12, dtype=np.uint8)
    moves = np.array([np.roll(solved, k) for k in (1, -1, 3, -3)])
    # A tie-rich deterministic Q head. All beam/sorting/collective/path code
    # is the vendored artifact. Also test an unreachable goal to see all layers.
    def model(_, x, dtype):
        return ((x[:, 0, None].astype(jnp.int32) + jnp.array([3,0,1,2])) % 5).astype(dtype)
    beam.model_apply = model
    compile_step = beam._compile_packed_step
    buffers = []
    def recording_compile(*args):
        step, lower_s, compile_s = compile_step(*args)
        def record(*carry):
            out = step(*carry)
            # Capture every retained state, packed backpointer and history
            # member before the next iteration donates these buffers.
            buffers.append([np.array(x, copy=True) for x in out])
            return out
        return record, lower_s, compile_s
    beam._compile_packed_step = recording_compile
    mesh8, mesh1 = beam.make_mesh(physical), beam.make_mesh(physical[:1])
    with tempfile.TemporaryDirectory() as tmp:
        for goal_type in ["unreachable", "exact", "endgame"]:
            init = np.roll(solved, 5 if goal_type=="endgame" else 6)
            if goal_type=="unreachable":
                init[[1, 2]] = init[[2, 1]]
            kwargs = dict(init_state_list=init.tolist(), v_params={}, all_moves=jnp.asarray(moves),
                          V0=jnp.asarray(solved), hash_vec=jnp.asarray(rng.integers(1, 10**9, 12)),
                          owner_hash_vec=jnp.asarray(rng.integers(1, 2**31, 12, dtype=np.uint32)),
                          B_local=16, K_per_peer=4, n_gen=4, state_size=12, num_steps=6,
                          internal_bs=2, history_depth=4, inv_move_tbl=jnp.array([1,0,3,2]),
                          tree_path=str(Path(tmp)/"tree.bin"), q_mode=True, reuse_compiled=False)
            if goal_type=="endgame":
                ztab = rng.integers(1,2**55,(12,12),dtype=np.int64)
                ball = np.vstack([solved, solved[moves]])
                eg = np.bitwise_xor.reduce(ztab[np.arange(12),ball],axis=1)
                kwargs.update(eg_hashes=jnp.asarray(np.sort(eg)), eg_ztab=jnp.asarray(ztab))
            for chunk in [None,4]:
                traces = []
                signatures = []
                placements = []
                for mesh, ranks in [(mesh8,None),(mesh1,8)]:
                    events = []
                    buffers.clear()
                    r = beam.beam_solve_v_only_spmd_packed(**kwargs, mesh=mesh, logical_ranks=ranks,
                                                          parent_chunk=chunk, on_layer=events.append)
                    placements.append(buffers.copy())
                    traces.append([(x["layer"],x["active_count"],x.get("rank_counts"))
                                   for x in events if x["event"]=="layer_end"])
                    fields = ("found","path_idx","found_pos_local","found_pos_rank",
                              "last_completed_step","min_v_trajectory_rank0")
                    signatures.append([r.get(key) for key in fields])
                assert traces[0]==traces[1] and signatures[0]==signatures[1], (traces,signatures)
                assert len(placements[0]) == len(placements[1]) > 0
                for distributed, local in zip(*placements):
                    for d, v in zip(distributed,local):
                        np.testing.assert_array_equal(d,v)
                assert signatures[0][0]==(goal_type!="unreachable") and traces[0], (traces,signatures)
                if goal_type!="unreachable":
                    trial = init
                    for m in signatures[0][1]:
                        trial = trial[moves[m]]
                    if goal_type=="endgame":
                        assert not np.array_equal(trial,solved)
                        suffix = next(move for move in moves if np.array_equal(trial[move],solved))
                        trial = trial[suffix]
                    np.testing.assert_array_equal(trial,solved)
                print(f"matched every JAX output, counts and replay: chunk={chunk}, "
                      f"target={goal_type}, completed={len(traces[0])}", flush=True)


if __name__ == "__main__":
    main()
