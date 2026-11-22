# Cube555 JAX solver source

The source snapshot accompanies [Andrey Lukyanenko's Kaggle notebook](https://www.kaggle.com/code/artgor/cayleypy-cube555-tpu-beam-q) in the [Cube555 TPU artifacts dataset](https://www.kaggle.com/datasets/artgor/cube555-tpu-artifacts). The local fetch receipt records 20 November 2026 at 17:40 UTC and notebook SHA-256 `0384177ed276f8bc1e64e2bd8cc519e5a903e9f3b1223db559160e67b345e304`.

Kaggle identifies the dataset as version 3, updated 11 November 2026, and lists its license as **CC0: Public Domain**. The two Python modules below came from that dataset. [Kaggle dataset metadata](https://www.kaggle.com/api/v1/datasets/view/artgor/cube555-tpu-artifacts) · [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)

| Original dataset file | SHA-256 of the supplied file |
| --- | --- |
| `jax_model.py` | `93955759e1c36031ab983a41249d10406d77eae52b9daee058421a2cde0e86d6` |
| `jax_beam_spmd_v_only.py` | `cb62b5e8319b44907114ca1feb06147024afd6c0e4bbc3af5c5fff84501f51f8` |

`jax_model.py` is copied byte-for-byte. The beam module uses a package-relative model import. Passing `logical_ranks=8` to its packed solver places the original eight rank functions and their all-to-all collective on one JAX device. In physical-rank mode the original shard map executes the same body on distinct devices. The source uses the JAX-version-appropriate spelling for marking a shard-map scan carry as varying. The logical-rank mode does not need that physical sharding annotation.

The optional `on_layer` callback receives start and end times in Unix seconds, plus the occupied frontier count at each completed layer. The seed uses up to 30 distinct choices replicated across eight ranks. Its event reports distinct choices in `active_count` and replicated occupancy in `active_slots`. Subsequent events also contain occupancy by receiver rank, plus measured device, host-transfer and backpointer-write times. A reported `path_idx` ends at the selected goal or endgame state. The calling notebook reconstructs the suffix from the endgame table and verifies the complete path on the original scramble.

Run `python -m tools.tpu_reference.parity_test` in a fresh process to compare physical and virtual rank placement on CPU.
