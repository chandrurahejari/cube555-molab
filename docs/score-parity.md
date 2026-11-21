# Reproducing Cube555 score parity

Run from the repository root with Python, NumPy, CPU JAX and PyTorch. The audit accepts the competition ZIP and a submission CSV containing valid paths. Put the published `jax_model.py`, `piece_layout_555.json`, `q555_f1_bell2k.pt` and `q555_2k_BEST.pt` together under the directory passed to `--artifacts`. The reference module and both checkpoints come from the [Cube555 TPU artifacts](https://www.kaggle.com/datasets/artgor/cube555-tpu-artifacts).

```sh
python tools/audit_score_parity.py --artifacts ./artifacts --competition ./competition.zip --submission ./submission.csv --output scores.json
```

For a GPU comparison on the Molab runtime, append `--torch-device cuda`. JAX defaults to CPU in both cases. The JSON records devices, library versions, file hashes, samples, per-head differences, action rankings and the native score-key conversion. The script exits unsuccessfully for FP32 differences above 0.0002 in any head or for nonfinite native scores.

## CPU measurement on November 22

The submission used here had score 100284. Every selected path was replayed against the competition goal. The supplied lengths were 23 (cube 57), 32 (68), 100 (142 and 166), and 110 (170, 256, 382 and 994). The test sampled each path at its start, one-third, two-thirds and six moves from the goal. It applied direct frames 0 and 3 and inverse frames 23 and 45. This gave 128 input states and 3,840 action scores for each head. The run used PyTorch 2.7.1 CPU, JAX 0.7.2 and NumPy 2.5.5.

| JAX / Torch precision | Head | Maximum absolute Q difference | Same first-ranked move out of 128 |
| --- | --- | ---: | ---: |
| FP32 / FP32 | Transformer | 0.000046 | 128 |
| FP32 / FP32 | ResMLP | 0.000111 | 128 |
| FP32 / FP32 | 0.8 / 0.2 blend | 0.000046 | 128 |
| BF16 / FP16 | Transformer | 0.1875 | 122 |
| BF16 / FP16 | ResMLP | 3.0313 | 108 |
| BF16 / FP16 | 0.8 / 0.2 blend | 0.7563 | 121 |

In FP32 all three scored outputs also agreed on the best three moves for all 128 states. In mixed precision, 126 of the 128 FP16 blend winners attained a minimum JAX BF16 score when ties were allowed. Ranking all 3,840 candidates together gave 116 shared candidates among the best 128. The native FP16 blend ranged from 4.8570 to 52.8938. All values were finite and within the native key range of 0 to 300. Rounding to units of 1/1024 preserved the first-ranked move on every sampled state.

These measurements check how both implementations rank identical verified states. Solving a 100- or 110-move cube still requires a complete beam search and replay of the returned path. The reported rankings use stable index ordering to break ties. Global candidate comparisons across samples measure sensitivity to score precision and do not represent a single search frontier.
