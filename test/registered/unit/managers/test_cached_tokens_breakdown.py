import unittest

from sglang.srt.managers.schedule_batch import Req


def make_req() -> Req:
    req = Req.__new__(Req)
    req.rid = "test-request"
    req.host_hit_length = 0
    req.storage_hit_length = 0
    req.cached_tokens = 0
    req.already_computed = 0
    req.cached_tokens_device = 0
    req.cached_tokens_host = 0
    req.cached_tokens_storage = 0
    req._cache_breakdown_computed = False
    return req


class TestCachedTokensBreakdown(unittest.TestCase):
    def assert_conserved(self, req: Req) -> None:
        self.assertEqual(
            req.cached_tokens,
            req.cached_tokens_device
            + req.cached_tokens_host
            + req.cached_tokens_storage,
        )

    def test_zero_pop_does_not_erase_storage_credit(self):
        req = make_req()

        req.record_storage_hit_tokens(34_304)
        req.record_storage_hit_tokens(0)

        self.assertEqual(req.storage_hit_length, 34_304)

    def test_later_chunk_consumes_retained_storage_credit(self):
        req = make_req()
        req.record_storage_hit_tokens(34_304)

        req.cached_tokens += 5_120
        req.account_cached_tokens_by_source(5_120)
        self.assertEqual(req.cached_tokens_device, 5_120)
        self.assertEqual(req.cached_tokens_storage, 0)
        self.assert_conserved(req)

        req.cached_tokens += 34_304
        req.account_cached_tokens_by_source(34_304)
        self.assertEqual(req.cached_tokens_device, 5_120)
        self.assertEqual(req.cached_tokens_host, 0)
        self.assertEqual(req.cached_tokens_storage, 34_304)
        self.assert_conserved(req)

    def test_first_chunk_splits_device_host_and_storage(self):
        req = make_req()
        req.host_hit_length = 60
        req.record_storage_hit_tokens(40)

        req.cached_tokens += 100
        req.account_cached_tokens_by_source(100)

        self.assertEqual(req.cached_tokens_device, 40)
        self.assertEqual(req.cached_tokens_host, 20)
        self.assertEqual(req.cached_tokens_storage, 40)
        self.assert_conserved(req)

    def test_tokens_after_storage_credit_are_device_hits(self):
        req = make_req()
        req.record_storage_hit_tokens(64)

        req.cached_tokens += 64
        req.account_cached_tokens_by_source(64)
        req.cached_tokens += 16
        req.account_cached_tokens_by_source(16)

        self.assertEqual(req.cached_tokens_device, 16)
        self.assertEqual(req.cached_tokens_storage, 64)
        self.assert_conserved(req)

    def test_rejects_negative_deltas(self):
        req = make_req()

        with self.assertRaisesRegex(RuntimeError, "cached token count regressed"):
            req.account_cached_tokens_by_source(-1)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            req.record_storage_hit_tokens(-1)


if __name__ == "__main__":
    unittest.main()
