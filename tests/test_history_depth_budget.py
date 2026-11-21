"""Check the preflight against the disk-first native whole-layer allocator."""
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUBE555_DEVICE_BYTES", str(96 << 30))
os.environ.setdefault("CUBE555_WORLD_SIZE", "1")
from tools.cayleypy_public import runner


class HistoryBudgetTests(unittest.TestCase):
    def test_capacity_matches_unsuccessful_search(self):
        for width, moves in [(1, 30), (64, 30), (256, 2), (64, 1)]:
            overhead = runner._HISTORY_SLOT_COUNT * width * (
                runner._CANDIDATE_META_BYTES + runner._HISTORY_ENTRY_BYTES)
            for ram_entries, disk_entries in [(4, 5), (12, 20), (64, 63), (31, 95), (400, 513)]:
                ram, disk = ram_entries, disk_entries
                current = 1
                layers = 0
                while True:
                    current = min(width, current * moves)
                    if disk >= current:
                        disk -= current
                    elif ram >= current:
                        ram -= current
                    else:
                        break
                    layers += 1
                for radius in (0, 5):
                    with self.subTest(width=width, moves=moves, ram=ram_entries, disk=disk_entries, radius=radius):
                        measured = runner.maximum_history_depth(SimpleNamespace(local_beam=width), moves, radius,
                            runner._MOLAB_RANKS * (overhead + ram_entries*runner._HISTORY_ENTRY_BYTES),
                            runner._MOLAB_RANKS * disk_entries*runner._HISTORY_ENTRY_BYTES)
                        self.assertEqual(layers, measured)
