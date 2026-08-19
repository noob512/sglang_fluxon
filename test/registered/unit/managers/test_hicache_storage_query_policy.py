"""Unit tests for the best-effort HiCache L3 lookup policy."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
)
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHiCacheStorageQueryPolicy(unittest.TestCase):
    def _l1_only_anchor(self):
        anchor = MagicMock()
        anchor.backuped = False
        anchor.get_last_hash_value.return_value = "l1-only-hash"
        return anchor

    def test_prefill_queries_l3_from_l1_only_non_root_without_metadata(self):
        root = object()
        anchor = self._l1_only_anchor()
        tree_cache = MagicMock()
        tree_cache.root_node = root
        tree_cache.hicache_storage_pass_prefix_keys = False
        scheduler = SimpleNamespace(
            enable_hicache_storage=True,
            tree_cache=tree_cache,
        )
        req = SimpleNamespace(
            rid="prefill-l1-only",
            prefix_indices=torch.tensor([0, 1]),
            host_hit_length=1,
            storage_metadata_hit_length=0,
            last_host_node=anchor,
            full_untruncated_fill_ids=list(range(8)),
            init_next_round_input=MagicMock(),
            _compute_max_prefix_len=MagicMock(return_value=7),
        )

        Scheduler._prefetch_kvcache(scheduler, req)

        self.assertIsNot(anchor, root)
        self.assertFalse(anchor.backuped)
        req.init_next_round_input.assert_called_once_with(tree_cache, cow_mamba=False)
        tree_cache.prefetch_from_storage.assert_called_once_with(
            req.rid,
            anchor,
            [3, 4, 5, 6],
            "l1-only-hash",
            None,
        )

    def test_decode_queries_l3_from_l1_only_non_root_without_metadata(self):
        root = object()
        anchor = self._l1_only_anchor()
        tree_cache = MagicMock()
        tree_cache.root_node = root
        tree_cache.hicache_storage_pass_prefix_keys = False
        tree_cache.query_storage_hit_length.return_value = 3
        queue = SimpleNamespace(
            scheduler=SimpleNamespace(enable_decode_hicache=True),
            tree_cache=tree_cache,
        )
        req = SimpleNamespace(origin_input_ids=list(range(8)))
        result = SimpleNamespace(
            device_indices=torch.tensor([0, 1]),
            host_hit_length=1,
            storage_metadata_hit_length=0,
            last_host_node=anchor,
            last_device_node=anchor,
        )

        match = DecodeHiCachePreallocMixin._build_decode_prefix_match(
            queue, req, result
        )

        self.assertIsNot(anchor, root)
        self.assertFalse(anchor.backuped)
        tree_cache.query_storage_hit_length.assert_called_once_with(
            anchor,
            [3, 4, 5, 6, 7],
            "l1-only-hash",
            None,
        )
        self.assertEqual(match.l3_storage_hit_length, 3)
        self.assertIs(match.last_host_node, anchor)


if __name__ == "__main__":
    unittest.main()
