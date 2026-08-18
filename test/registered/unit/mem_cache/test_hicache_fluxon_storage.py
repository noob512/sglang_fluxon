"""Unit tests for the Fluxon HiCache adapter using an in-memory fake store."""

import os
import unittest
from copy import deepcopy
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.storage.fluxon.hicache_fluxon import HiCacheFluxon
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Result:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    def is_ok(self):
        return self.error is None

    def unwrap(self):
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.value

    def unwrap_error(self):
        if self.error is None:
            raise RuntimeError("result does not contain an error")
        return self.error


class _FakeClientConfig:
    loaded_paths = []

    def __init__(self, config):
        self._config = deepcopy(config)
        self.instance_key = self._config["instance_key"]

    @classmethod
    def from_file(cls, path):
        cls.loaded_paths.append(path)
        return cls(
            {
                "instance_key": "fake-fluxon",
                "contribute_to_cluster_pool_size": {"dram": 1, "vram": {}},
            }
        )

    def to_dict(self):
        return deepcopy(self._config)


class _FakePutOptionalArgs:
    def __init__(
        self,
        lease_id=None,
        reject_if_inflight_same_key=False,
        reject_if_exist_same_key=False,
        write_through=True,
        make_replica_task=True,
        make_replica_task_mask=None,
        radix_parent_keys=None,
        content_depths=None,
        atomic_group_lens=None,
    ):
        self.lease_id = lease_id
        self.reject_if_inflight_same_key = reject_if_inflight_same_key
        self.reject_if_exist_same_key = reject_if_exist_same_key
        self.write_through = write_through
        self.make_replica_task = make_replica_task
        self.make_replica_task_mask = make_replica_task_mask
        self.radix_parent_keys = radix_parent_keys
        self.content_depths = content_depths
        self.atomic_group_lens = atomic_group_lens


class _FakeStore:
    def __init__(self):
        self.values = {}
        self.closed = False

    def batch_is_exist(self, keys, pin_ttl=None):
        return [int(key in self.values) for key in keys]

    def is_exist(self, key):
        return _Result(key in self.values)

    def put_blocking(self, key, value, opts=None):
        self.values[key] = {
            name: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in value.items()
        }
        return _Result(None)

    def get_blocking(self, key):
        if key not in self.values:
            return _Result(error=f"missing key: {key}")
        return _Result(self.values[key])

    def batch_get_blocking(self, keys, concurrency=None):
        return [
            _Result(self.values[key])
            if key in self.values
            else _Result(error=f"missing key: {key}")
            for key in keys
        ]

    def batch_put_blocking(self, keys, values, opts=None, concurrency=None):
        return [
            self.put_blocking(key, value, opts=opts) for key, value in zip(keys, values)
        ]

    def close(self):
        self.closed = True
        return _Result(None)


class _FakeHostPool:
    def __init__(self, data, page_size=2):
        self.data = data.contiguous()
        self.page_size = page_size

    def get_page_buffer_meta(self, indices):
        indices = [int(index) for index in indices.tolist()]
        ptrs = []
        sizes = []
        for offset in range(0, len(indices), self.page_size):
            page_indices = indices[offset : offset + self.page_size]
            first = page_indices[0]
            assert page_indices == list(range(first, first + self.page_size))
            page = self.data[first : first + self.page_size]
            ptrs.append(page.data_ptr())
            sizes.append(page.numel() * page.element_size())
        return ptrs, sizes

    def get_dummy_flat_data_page(self):
        return torch.zeros_like(self.data[: self.page_size]).reshape(-1)

    def get_data_page(self, index, flat=True):
        page = self.data[int(index) : int(index) + self.page_size]
        return page.reshape(-1) if flat else page

    def set_from_flat_data_page(self, index, data_page):
        page = self.data[int(index) : int(index) + self.page_size]
        page.copy_(data_page.reshape_as(page))


def _storage_config(extra_config):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="fake-model",
        extra_config=extra_config,
    )


class TestHiCacheFluxon(unittest.TestCase):
    def _make_backend(self):
        store = _FakeStore()
        with patch(
            "sglang.srt.mem_cache.storage.fluxon.hicache_fluxon._import_fluxon_symbols",
            return_value=(
                _FakeClientConfig,
                lambda _config: _Result(store),
                _FakePutOptionalArgs,
                object,
            ),
        ):
            backend = HiCacheFluxon(
                _storage_config({"config_path": "/fake/fluxon.yaml"})
            )
        self.addCleanup(backend.close)
        return backend, store

    def test_factory_registers_fluxon(self):
        entry = StorageBackendFactory._registry["fluxon"]
        self.assertEqual(
            entry["module_path"],
            "sglang.srt.mem_cache.storage.fluxon.hicache_fluxon",
        )
        self.assertEqual(entry["class_name"], "HiCacheFluxon")

    def test_config_path_does_not_fall_back_to_environment(self):
        with patch.dict(
            os.environ,
            {"SGLANG_REMOTE_PAGE_CACHE_FLUXON_CONFIG_PATH": "/legacy.yaml"},
        ):
            with self.assertRaisesRegex(RuntimeError, r"extra_config\['config_path'\]"):
                HiCacheFluxon(_storage_config({}))

    def test_rejects_parallel_fluxon_config_fields(self):
        with self.assertRaisesRegex(ValueError, "only accepts 'config_path'"):
            HiCacheFluxon(
                _storage_config(
                    {
                        "config_path": "/fake/fluxon.yaml",
                        "batch_concurrency": 32,
                    }
                )
            )

    def test_exists_get_set_and_close(self):
        backend, store = self._make_backend()
        value = torch.arange(8, dtype=torch.float32)

        self.assertTrue(backend.set("present", value=value))
        self.assertTrue(backend.exists("present"))
        self.assertFalse(backend.exists("missing"))
        self.assertEqual(backend.batch_exists(["present", "missing", "present"]), 1)

        target = torch.zeros_like(value)
        result = backend.get("present", target_location=target)
        self.assertIs(result, target)
        self.assertTrue(torch.equal(result, value))

        backend.close()
        self.assertTrue(store.closed)
        self.assertIsNone(backend.store)

    def test_batch_v1_round_trip(self):
        backend, _ = self._make_backend()
        keys = ["page-0", "page-1"]
        indices = torch.arange(4, dtype=torch.int64)
        source = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
        source_pool = _FakeHostPool(source)
        backend.register_mem_pool_host(source_pool)

        self.assertEqual(backend.batch_set_v1(keys, indices), [True, True])

        destination = torch.zeros_like(source)
        destination_pool = _FakeHostPool(destination)
        backend.register_mem_pool_host(destination_pool)
        self.assertEqual(backend.batch_get_v1(keys, indices), [True, True])
        self.assertTrue(torch.equal(destination_pool.data, source))


if __name__ == "__main__":
    unittest.main()
