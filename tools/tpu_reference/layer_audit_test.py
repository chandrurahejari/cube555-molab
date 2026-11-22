"""Exercise full-slot capture, order-independent comparison and packed parent diagnostics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .layer_audit import compare_directories, record_layer


class RankOnly:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.calls = []

    def __getitem__(self, rank):
        assert isinstance(rank, int)
        self.calls.append(rank)
        return self.array[rank]

    def __array__(self, *args):
        raise AssertionError("Attempted to copy all eight ranks to the host at once.")


class LayerAuditTest(unittest.TestCase):
    def test_snapshots_and_first_difference(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp)/"left", Path(temp)/"right"
            states = np.zeros((8, 4, 150), dtype=np.uint8)
            states[0, 0] = states[0, 2] = np.arange(150)
            states[0, 1] = states[0, 0, ::-1]
            packed = np.arange(32, dtype=np.uint32).reshape(8, 4)
            scores = np.arange(32, dtype=np.float32).reshape(8, 4)
            scores[7, 0] = np.nan
            boxed = RankOnly(states)
            a = record_layer(left, 0, boxed, None, scores, save_arrays=True)
            self.assertEqual(boxed.calls, list(range(8)))
            self.assertEqual(a["active_count"], 3)
            record_layer(right, 0, states, None, scores, True)
            with np.load(left/"layer-0000.npz") as seed:
                self.assertEqual(set(seed.files), {"states", "scores"})
                np.testing.assert_array_equal(seed["states"], states)
                np.testing.assert_array_equal(seed["scores"], scores)
            self.assertTrue(compare_directories(left, right)["equal"])
            a = record_layer(left, 1, states, packed, scores, True)
            swapped = states.copy()
            swapped[0, [0, 1]] = swapped[0, [1, 0]]
            b = record_layer(right, 1, swapped, packed, scores, True)
            self.assertNotEqual(a["ranks"][0]["states_sha256"], b["ranks"][0]["states_sha256"])
            self.assertEqual(a["ranks"][0]["sorted_fingerprints_sha256"], b["ranks"][0]["sorted_fingerprints_sha256"])
            diff = compare_directories(left, right)
            self.assertEqual(diff["first_unequal"], 1)
            self.assertTrue(diff["difference"]["same_global_multiset"])
            changed = states.copy()
            changed[0, 2] = changed[0, 1]
            parents = packed.copy()
            parents[0, 2] = (12 << 27) + (3 << 24) + 8
            scores2 = scores.copy()
            scores2[0, 0] = 0.5
            record_layer(right, 1, changed, parents, scores2, True)
            diff = compare_directories(left, right)["difference"]
            self.assertEqual(diff["left_only_count"], 1)
            self.assertEqual(diff["right_only_count"], 1)
            self.assertEqual(diff["packed"]["first_right"]["backpointer"]["parent_rank"], 3)
            self.assertEqual(diff["scores"]["max_abs_on_same_stickers"], 0.5)
            self.assertEqual([json.loads(line)["layer"] for line in (left/"layers.jsonl").read_text().splitlines()], [0, 1])


if __name__ == "__main__":
    unittest.main()
