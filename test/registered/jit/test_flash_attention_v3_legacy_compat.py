import unittest

from sglang.jit_kernel.flash_attention_v3 import _call_fa3_kernel


class TestFlashAttentionV3LegacyCompat(unittest.TestCase):
    def test_disabled_only_qv_is_omitted_for_legacy_kernel(self):
        def legacy_kernel(value):
            return value + 1

        self.assertEqual(
            _call_fa3_kernel(legacy_kernel, 4, only_qv=False),
            5,
        )

    def test_enabled_only_qv_remains_fail_closed(self):
        def legacy_kernel(value):
            return value + 1

        with self.assertRaisesRegex(TypeError, "only_qv"):
            _call_fa3_kernel(legacy_kernel, 4, only_qv=True)

    def test_legacy_only_qv_and_out_fallbacks_compose(self):
        def legacy_kernel(value):
            return value + 1

        self.assertEqual(
            _call_fa3_kernel(
                legacy_kernel,
                4,
                only_qv=False,
                out=object(),
            ),
            5,
        )


if __name__ == "__main__":
    unittest.main()
