"""Compare streamed and unchunked Kaggle steps using the production BF16 blend.

This is a small in-memory diagnostic. It retains every layer for exact comparison.
"""
import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--competition", type=Path, required=True)
    parser.add_argument("--cube", type=int, default=76)
    parser.add_argument("--local-width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--micro", type=int, default=4)
    parser.add_argument("--platform", choices=["cpu","gpu"], default="cpu")
    args = parser.parse_args()
    local = args.local_width
    if local < 4 or local % 4 or local % args.chunk or args.chunk % args.micro:
        parser.error("local width must divide into four peer-quota units and integral chunks/microbatches")
    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"
    import jax
    jax.config.update("jax_enable_x64",True)
    import jax.numpy as jnp
    from . import jax_beam_spmd_v_only as beam
    from . import jax_model
    from tools.tpu_solve import PINNED, read_puzzle, verify
    args.pid = args.cube
    start, goal, moves, names = read_puzzle(args)
    layout = args.assets / "piece_layout_555.json"
    weights = ["q555_f1_bell2k.pt","q555_2k_BEST.pt"]
    for name in [*weights,layout.name]:
        verify(args.assets/name,PINNED[name])
    blend = jax_model.make_blend([jax_model.load_params_from_pt(args.assets/name,layout)
                                  for name in weights],[0.8,0.2])
    owner = np.random.default_rng(12345).integers(0,2**32-1,150,dtype=np.uint32)
    h = np.random.default_rng(0).integers(0,10**15,150,dtype=np.int64)
    inverse = [names.index(n[1:] if n.startswith("-") else "-"+n) for n in names]
    mesh = beam.make_mesh([jax.devices(args.platform)[0]])
    inputs = dict(init_state_list=start.tolist(),v_params=blend,all_moves=jnp.asarray(moves),
                  V0=jnp.asarray(goal),hash_vec=jnp.asarray(h),mesh=mesh,B_local=local,
                  K_per_peer=2*local//8,n_gen=30,state_size=150,num_steps=args.layers,
                  dtype=jnp.bfloat16,internal_bs=args.micro,owner_hash_vec=jnp.asarray(owner),
                  history_depth=4,history_exact=True,inv_move_tbl=jnp.asarray(inverse),
                  pack_v_score=True,q_mode=True,logical_ranks=8,reuse_compiled=False)
    root_q = np.asarray(jax_model.apply(blend,start[None],jnp.bfloat16)).reshape(-1)
    root_ties = len(root_q) - len(np.unique(root_q))
    print(f"cube={args.cube} width={8*local} root_tied_actions={root_ties}",flush=True)
    compile_step = beam._compile_packed_step
    score_rank = jax.jit(lambda x: jax_model.apply(blend,x,jnp.bfloat16))
    labels = ("frontier","backpointers","min_q","found_step","found_pos","found_rank","verify_state",
              "history","incoming_move","trace_hit","trace_pos","trace_q","trace_cutoff","rank_occupancy")
    baseline = []
    counts = []
    mode = None
    def instrument(*parts):
        step, lower, compile_time = compile_step(*parts)
        def capture(*carry):
            layer = int(carry[-1])+1
            out = step(*carry)
            snapshot = [np.array(x,copy=True) for x in out]
            if mode is None:
                baseline.append(snapshot)
                frontier = snapshot[0].reshape(-1,150)
                real = frontier[np.any(frontier,axis=1)]
                children = real[:,moves].reshape(-1,150)
                n_unique = len(np.unique(real,axis=0))
                if n_unique != len(real):
                    raise AssertionError(f"Masked duplicate states re-entered layer {layer}: {len(real)-n_unique}")
                child_states = np.unique(children,axis=0)
                child_h = child_states.astype(np.int64) @ h
                if len(np.unique(child_h)) != len(child_states):
                    raise AssertionError(f"distinct child states share reference hash at depth {layer}")
                q = np.asarray(score_rank(snapshot[0][0]))
                rank_q = q[np.any(snapshot[0][0],axis=1)].reshape(-1)
                tied = len(rank_q)-len(np.unique(rank_q))
                duplicate_children = len(children)-len(child_states)
                counts.append((len(real),n_unique,duplicate_children,tied))
                sig = hashlib.sha256(snapshot[0].tobytes()).hexdigest()[:16]
                print(f"depth={layer} occupied={len(real)} distinct={n_unique} "
                      f"rank0_tied_q={tied}/{len(rank_q)} next_duplicate_children={duplicate_children} "
                      f"child_hash_collisions=0 frontier={sig}",flush=True)
            else:
                for label,want,got in zip(labels,baseline[layer-2],snapshot):
                    if not np.array_equal(want,got):
                        raise AssertionError(f"chunk={mode} depth={layer} tensor={label} "
                                             f"differing_entries={np.count_nonzero(want!=got)}")
                print(f"chunk={mode} depth={layer} 14/14 tensors identical",flush=True)
            return out
        return capture,lower,compile_time
    beam._compile_packed_step = instrument
    with tempfile.TemporaryDirectory() as folder:
        for mode in [None,args.chunk]:
            solution = beam.beam_solve_v_only_spmd_packed(**inputs,parent_chunk=mode,
                tree_path=str(Path(folder)/"backpointers.bin"))
            if solution["found"]:
                raise RuntimeError("Choose a harder cube to exercise full layers")
    if not any(distinct==8*local and dup>0 and ties>0 for _,distinct,dup,ties in counts):
        raise AssertionError("No distinct saturated beam with ties and duplicate children; increase depth")
    if len(counts) != args.layers-1:
        raise AssertionError("Incomplete sample")


if __name__=="__main__":
    main()
