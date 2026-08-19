# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import ctypes
import hashlib
import inspect
import importlib
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
    STORAGE_BATCH_SIZE,
)
from sglang.srt.observability.metrics_collector import StorageMetrics

logger = logging.getLogger(__name__)

_CUDA_HOST_REGISTER_PORTABLE = 1
_CUDA_HOST_REGISTER_MAPPED = 2
_FLUXON_PLAN_BLOB_MAGIC = 0x4658504C414E5631
_CUDA_HOST_REGISTER_READ_ONLY = 8
_E44_BATCH_CONCURRENCY = 32
_E44_BATCH_EXISTS_PIN_TTL_MS = 1_200
_E44_WARM_PENDING_LIMIT = 4_096
_E44_WARM_SUBMIT_LIMIT = 128
_E44_WARM_DRAIN_BUDGET = 64
_REGISTERED_FLUXON_SEGMENTS: set[tuple[int, int, int]] = set()
_REGISTERED_FLUXON_SEGMENTS_LOCK = threading.Lock()
_REPLICA_TASK_POLICIES = frozenset(
    (
        "eager_all",
        "prefix_root_only",
        "prefix_depth_ratio",
        "prefix_end_depth_ratio",
        "ratio_only",
        "kv_score_only",
        "kv_score_low_only",
        "kv_score_pressure_budget",
    )
)


@dataclass(frozen=True)
class _FluxonReplicaTaskAdmissionConfig:
    policy: str
    score_threshold: float
    score_bypass_threshold: float
    admission_ratio: float
    min_replica_pages: int
    max_replica_pages_per_batch: int


@dataclass(frozen=True)
class _FluxonReplicaTaskConfig:
    enabled: bool
    admission: _FluxonReplicaTaskAdmissionConfig
    metrics_sample_interval_ms: int


class _FluxonWarmOkResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def is_ok(self) -> bool:
        return True

    def unwrap(self) -> Any:
        return self._value


class _FluxonWarmErrorResult:
    def __init__(self, error: Any) -> None:
        self._error = error

    def is_ok(self) -> bool:
        return False

    def unwrap_error(self) -> Any:
        return self._error


class _FluxonBatchWarmFuture:
    def __init__(
        self,
        batch_future: Future,
        batch_index: int,
        storage_key: str,
    ) -> None:
        self._batch_future = batch_future
        self._batch_index = batch_index
        self._storage_key = storage_key

    def is_waiting(self) -> bool:
        batch_future = self._batch_future
        return batch_future is not None and not batch_future.done()

    def wait(self) -> Any:
        batch_future = self._batch_future
        if batch_future is None:
            return _FluxonWarmErrorResult(
                f"batch warm result already consumed for {self._storage_key}"
            )
        try:
            batch_results = batch_future.result()
        except Exception as exc:
            self._batch_future = None
            return _FluxonWarmErrorResult(exc)
        self._batch_future = None

        if hasattr(batch_results, "is_ok"):
            if not batch_results.is_ok():
                return _FluxonWarmErrorResult(batch_results.unwrap_error())
            batch_results = batch_results.unwrap()

        if not isinstance(batch_results, list):
            return _FluxonWarmErrorResult(
                f"batch warm returned non-list payload for {self._storage_key}: "
                f"{type(batch_results)}"
            )
        if self._batch_index >= len(batch_results):
            return _FluxonWarmErrorResult(
                f"batch warm returned too few results for {self._storage_key}: "
                f"index={self._batch_index} results={len(batch_results)}"
            )

        item = batch_results[self._batch_index]
        # batch_get_blocking returns one MemHolder per key. Clear this slot so a
        # shared batch future does not keep the whole batch alive after this key
        # is drained or consumed by the foreground path.
        batch_results[self._batch_index] = None
        if hasattr(item, "is_ok"):
            return item
        if item is None:
            return _FluxonWarmErrorResult(
                f"batch warm returned None for {self._storage_key}"
            )
        return _FluxonWarmOkResult(item)


class _FluxonMergedBatchRetCodeFuture:
    def __init__(
        self,
        op_name: str,
        total_items: int,
        group_futures: list[tuple[list[int], Any]],
    ) -> None:
        self._op_name = op_name
        self._total_items = total_items
        self._group_futures = group_futures

    def is_waiting(self) -> bool:
        for _, future in self._group_futures:
            is_waiting = getattr(future, "is_waiting", None)
            if is_waiting is None or bool(is_waiting()):
                return True
        return False

    def wait(self) -> Any:
        result_codes: list[int | None] = [None] * self._total_items
        first_error: Any | None = None

        for page_indices, future in self._group_futures:
            try:
                wait_result = future.wait()
            except Exception as exc:
                if first_error is None:
                    first_error = _FluxonWarmErrorResult(
                        f"Fluxon {self._op_name} group wait raised: {exc}"
                    )
                continue
            if not hasattr(wait_result, "is_ok"):
                if first_error is None:
                    first_error = _FluxonWarmErrorResult(
                        f"Fluxon {self._op_name} group future returned invalid result object"
                    )
                continue
            if not wait_result.is_ok():
                if first_error is None:
                    first_error = wait_result
                continue

            raw_codes = wait_result.unwrap()
            if not isinstance(raw_codes, list) or len(raw_codes) != len(page_indices):
                if first_error is None:
                    first_error = _FluxonWarmErrorResult(
                        f"Fluxon {self._op_name} group returned mismatched ret-code "
                        f"list: expected={len(page_indices)} got={raw_codes!r}"
                    )
                continue
            for page_index, code in zip(page_indices, raw_codes):
                if not isinstance(code, int):
                    if first_error is None:
                        first_error = _FluxonWarmErrorResult(
                            f"Fluxon {self._op_name} group returned non-int ret-code: "
                            f"{type(code)}"
                        )
                    continue
                result_codes[page_index] = int(code)

        if first_error is not None:
            return first_error
        if any(code is None for code in result_codes):
            return _FluxonWarmErrorResult(
                f"Fluxon {self._op_name} did not populate all ret-code slots"
            )
        return _FluxonWarmOkResult([int(code) for code in result_codes])


def _wait_submitted_replica_groups(
    op_name: str,
    group_futures: list[tuple[list[int], Any]],
) -> None:
    for _, future in group_futures:
        try:
            wait_result = future.wait()
        except Exception as exc:
            logger.warning(
                "Fluxon %s cleanup wait raised: %s",
                op_name,
                exc,
            )
            continue
        if not hasattr(wait_result, "is_ok"):
            logger.warning(
                "Fluxon %s cleanup wait returned invalid result object",
                op_name,
            )
            continue
        if not wait_result.is_ok():
            logger.warning(
                "Fluxon %s cleanup wait failed: %s",
                op_name,
                wait_result.unwrap_error(),
            )


class _FluxonFragmentPointerView:
    def __init__(
        self,
        fragment_ptrs: list[int],
        fragment_lens: list[int],
        keepalive: Any,
    ) -> None:
        self.fragment_ptrs = fragment_ptrs
        self.fragment_lens = fragment_lens
        self.keepalive = keepalive
        self.total_bytes = sum(fragment_lens)


class _FluxonGpuStagingLease:
    def __init__(self, pool: "_FluxonGpuStagingPool", slot_indices: list[int]) -> None:
        self._pool = pool
        self._slot_indices = slot_indices
        self._initial_page_count = len(slot_indices)
        self._acquired_at = time.monotonic()
        self._released = False
        self.destinations = tuple(
            pool.registration.destination(
                pool.base_ptr + slot_index * pool.slot_size,
                pool.slot_size,
            )
            for slot_index in slot_indices
        )
        self.value_ptrs = tuple(destination.ptr for destination in self.destinations)

    @property
    def page_count(self) -> int:
        return len(self._slot_indices)

    @property
    def slot_size(self) -> int:
        return self._pool.slot_size

    @property
    def released(self) -> bool:
        return self._released

    def trim_after_transfer(self, page_count: int) -> None:
        if self._released:
            raise RuntimeError("GPU staging lease is already released")
        if page_count < 0 or page_count > len(self._slot_indices):
            raise ValueError(
                "GPU staging trim is outside the lease: "
                f"trim={page_count} slots={len(self._slot_indices)}"
            )
        tail = self._slot_indices[page_count:]
        if tail:
            self._pool._release_slots(tail)
            del self._slot_indices[page_count:]
            self.destinations = self.destinations[:page_count]
            self.value_ptrs = self.value_ptrs[:page_count]

    def release(self, reason: str = "unspecified") -> None:
        if self._released:
            return
        self._released = True
        slots = self._slot_indices
        self._slot_indices = []
        self.destinations = ()
        self.value_ptrs = ()
        self._pool._release_lease(
            slots,
            initial_page_count=self._initial_page_count,
            acquired_at=self._acquired_at,
            reason=str(reason),
        )


class _FluxonGpuStagingPool:
    def __init__(
        self,
        store: Any,
        slot_size: int,
        slot_count: int,
        device_id: int,
        allocator_type: Any,
    ) -> None:
        if slot_size <= 0 or slot_count <= 0:
            raise ValueError(
                f"GPU staging requires positive geometry: size={slot_size} count={slot_count}"
            )
        self.store = store
        self.slot_size = int(slot_size)
        self.slot_count = int(slot_count)
        self.device_id = int(device_id)
        self._lock = threading.Lock()
        self._allocator = allocator_type(self.slot_count)
        self._closed = False
        self._active_leases = 0
        self._high_watermark_slots = 0
        self._admission_attempts = 0
        self._admission_requested_pages = 0
        self._admission_reasons: dict[str, int] = {}
        self._lease_releases = 0
        self._lease_hold_ms_sum = 0.0
        self._lease_hold_ms_max = 0.0
        self._release_reasons: dict[str, int] = {}
        with torch.cuda.device(self.device_id):
            self.tensor = torch.empty(
                self.slot_size * self.slot_count,
                dtype=torch.uint8,
                device=torch.device("cuda", self.device_id),
            )
        self.base_ptr = int(self.tensor.data_ptr())
        register_result = store.register_gpu_buffer(
            self.base_ptr,
            int(self.tensor.numel()),
            self.device_id,
        )
        if not register_result.is_ok():
            self.tensor = None
            raise RuntimeError(
                f"Fluxon GPU staging registration failed: {register_result.unwrap_error()}"
            )
        self.registration = register_result.unwrap()
        logger.info(
            "Fluxon GPU staging registered: registration_id=%d device=%d "
            "base=%#x slot_size=%d slots=%d bytes=%d",
            self.registration.registration_id,
            self.device_id,
            self.base_ptr,
            self.slot_size,
            self.slot_count,
            int(self.tensor.numel()),
        )

    def _snapshot_locked(self) -> dict[str, int]:
        free_slots = int(self._allocator.free_count)
        return {
            "capacity_slots": self.slot_count,
            "free_slots": free_slots,
            "live_slots": self.slot_count - free_slots,
            "active_leases": self._active_leases,
            "high_watermark_slots": self._high_watermark_slots,
        }

    def try_reserve(
        self,
        page_count: int,
        admission_block_reason: Optional[str] = None,
    ) -> tuple[Optional[_FluxonGpuStagingLease], dict[str, Any]]:
        requested_pages = int(page_count)
        slots: list[int] | None = None
        with self._lock:
            before = self._snapshot_locked()
            if admission_block_reason is not None:
                reason = str(admission_block_reason)
            elif self._closed:
                reason = "pool_closed"
            elif requested_pages <= 0:
                reason = "not_eligible"
            elif requested_pages > self.slot_count:
                reason = "request_exceeds_capacity"
            else:
                slots = self._allocator.try_reserve(requested_pages)
                if slots is None:
                    reason = "insufficient_free_slots"
                else:
                    reason = "selected"
                    self._active_leases += 1
                    self._high_watermark_slots = max(
                        self._high_watermark_slots,
                        int(self._allocator.live_count),
                    )
            self._admission_attempts += 1
            self._admission_requested_pages += max(requested_pages, 0)
            self._admission_reasons[reason] = (
                self._admission_reasons.get(reason, 0) + 1
            )
            after = self._snapshot_locked()
            admission = {
                "reason": reason,
                "requested_pages": requested_pages,
                "capacity_slots": before["capacity_slots"],
                "free_slots_before": before["free_slots"],
                "live_slots_before": before["live_slots"],
                "active_leases_before": before["active_leases"],
                "free_slots_after": after["free_slots"],
                "live_slots_after": after["live_slots"],
                "active_leases_after": after["active_leases"],
                "high_watermark_slots": after["high_watermark_slots"],
            }
        if slots is None:
            return None, admission
        try:
            return _FluxonGpuStagingLease(self, slots), admission
        except Exception:
            with self._lock:
                self._allocator.release(slots)
                self._active_leases -= 1
            raise

    def _release_slots(self, slots: list[int]) -> None:
        if not slots:
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU staging slots released after pool close")
            self._allocator.release(slots)

    def _release_lease(
        self,
        slots: list[int],
        initial_page_count: int,
        acquired_at: float,
        reason: str,
    ) -> None:
        held_ms = (time.monotonic() - acquired_at) * 1000.0
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU staging lease released after pool close")
            if self._active_leases <= 0:
                raise RuntimeError("GPU staging active lease counter underflow")
            self._allocator.release(slots)
            self._active_leases -= 1
            self._lease_releases += 1
            self._lease_hold_ms_sum += held_ms
            self._lease_hold_ms_max = max(self._lease_hold_ms_max, held_ms)
            self._release_reasons[reason] = self._release_reasons.get(reason, 0) + 1
            snapshot = self._snapshot_locked()
        logger.info(
            "Fluxon GPU staging lease released: reason=%s held_ms=%.3f "
            "initial_slots=%d released_slots=%d free_slots=%d live_slots=%d "
            "active_leases=%d high_watermark_slots=%d",
            reason,
            held_ms,
            initial_page_count,
            len(slots),
            snapshot["free_slots"],
            snapshot["live_slots"],
            snapshot["active_leases"],
            snapshot["high_watermark_slots"],
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            live_slots = int(self._allocator.live_count)
            if live_slots != 0 or self._active_leases != 0:
                raise RuntimeError(
                    "GPU staging close has live allocations: "
                    f"slots={live_slots}/{self.slot_count} "
                    f"leases={self._active_leases}"
                )
            self._closed = True
            snapshot = self._snapshot_locked()
            admission_attempts = self._admission_attempts
            admission_requested_pages = self._admission_requested_pages
            admission_reasons = dict(self._admission_reasons)
            lease_releases = self._lease_releases
            lease_hold_ms_sum = self._lease_hold_ms_sum
            lease_hold_ms_max = self._lease_hold_ms_max
            release_reasons = dict(self._release_reasons)
        unregister_result = self.store.unregister_gpu_buffer(self.registration)
        if not unregister_result.is_ok():
            with self._lock:
                self._closed = False
            raise RuntimeError(
                f"Fluxon GPU staging unregister failed: {unregister_result.unwrap_error()}"
            )
        _ = unregister_result.unwrap()
        logger.info(
            "Fluxon GPU staging pool Snapshot: device=%d capacity_slots=%d "
            "free_slots=%d live_slots=%d active_leases=%d "
            "high_watermark_slots=%d admission_attempts=%d "
            "admission_requested_pages=%d admission_reasons=%s "
            "lease_releases=%d lease_hold_ms_avg=%.3f lease_hold_ms_max=%.3f "
            "release_reasons=%s",
            self.device_id,
            snapshot["capacity_slots"],
            snapshot["free_slots"],
            snapshot["live_slots"],
            snapshot["active_leases"],
            snapshot["high_watermark_slots"],
            admission_attempts,
            admission_requested_pages,
            admission_reasons,
            lease_releases,
            lease_hold_ms_sum / lease_releases if lease_releases else 0.0,
            lease_hold_ms_max,
            release_reasons,
        )
        self.registration = None
        self.tensor = None


def _import_fluxon_symbols() -> tuple[Any, Any, Any, Any]:
    fluxon_mod = importlib.import_module("fluxon_py")
    kvclient_interface_mod = importlib.import_module("fluxon_py.kvclient.kvclient_interface")
    pyo3_tool_mod = importlib.import_module("fluxon_py.tool")
    fluxon_pyo3_mod = pyo3_tool_mod.import_fluxon_pyo3_local()
    return (
        fluxon_mod.FluxonKvClientConfig,
        fluxon_mod.new_store,
        kvclient_interface_mod.PutOptionalArgs,
        fluxon_pyo3_mod.FixedSlabAllocator,
    )


def _build_put_optional_args(put_optional_args_type: Any, **kwargs: Any) -> Any:
    try:
        supported_args = set(inspect.signature(put_optional_args_type).parameters)
    except (TypeError, ValueError):
        supported_args = set(kwargs)
    return put_optional_args_type(
        **{key: value for key, value in kwargs.items() if key in supported_args}
    )


def _method_supported_kwargs(method: Any) -> tuple[set[str], bool] | None:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return None
    supported: set[str] = set()
    accepts_var_keyword = False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_keyword = True
        elif parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            supported.add(parameter.name)
    return supported, accepts_var_keyword


def _is_unexpected_fluxon_put_kwarg(exc: TypeError) -> bool:
    message = str(exc)
    return (
        "local_fast_put_start" in message
        and "unexpected keyword argument" in message
        and (
            "'write_through'" in message
            or "'make_replica_task'" in message
            or "'make_replica_task_mask'" in message
            or "'radix_parent_keys'" in message
            or "'content_depths'" in message
            or "'atomic_group_lens'" in message
        )
    )


def _is_zero_contribution_config(config_dict: dict[str, Any]) -> bool:
    contrib = config_dict.get("contribute_to_cluster_pool_size")
    if contrib is None:
        return True
    if not isinstance(contrib, dict):
        return False
    dram = int(contrib.get("dram", 0))
    vram = contrib.get("vram", {})
    if not isinstance(vram, dict):
        return False
    return dram == 0 and all(int(size) == 0 for size in vram.values())


def _extra_config_bool(
    extra_config: dict[str, Any], key: str, default: bool = False
) -> bool:
    value = extra_config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _as_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _as_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a float")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {parsed}")
    return parsed


def _as_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0, got {parsed}")
    return parsed


def _optional_positive_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    if isinstance(value, bool):
        return default if value else None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return default
    parsed = int(value)
    return parsed if parsed > 0 else None


def _namespace_config_for_process(config: Any, config_dict: dict[str, Any]) -> dict[str, Any]:
    pid = os.getpid()
    config_dict["instance_key"] = f"{config.instance_key}_pid{pid}"

    if not _is_zero_contribution_config(config_dict):
        fluxonkv_spec = config_dict.get("fluxonkv_spec")
        if isinstance(fluxonkv_spec, dict):
            for field in ("shared_memory_path", "shared_file_path"):
                path = fluxonkv_spec.get(field)
                if path:
                    fluxonkv_spec[field] = os.path.join(str(path), f"pid{pid}")

    return config_dict


def _reject_unknown_fields(
    config: dict[str, Any], allowed_fields: set[str], field_name: str
) -> None:
    unknown = sorted(set(config) - allowed_fields)
    if unknown:
        raise ValueError(f"{field_name} has unknown field(s): {unknown}")


def _parse_replica_task_config(
    extra_config: dict[str, Any],
) -> _FluxonReplicaTaskConfig:
    if "remote_publish" in extra_config:
        raise ValueError(
            "remote_publish was removed from Fluxon HiCache config; use replica_task"
        )
    raw_config = extra_config.get("replica_task")
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError("replica_task must be a dict")

    _reject_unknown_fields(
        raw_config,
        {"enabled", "admission", "metrics_sample_interval_ms"},
        "replica_task",
    )

    raw_admission = raw_config.get("admission", {})
    if not isinstance(raw_admission, dict):
        raise ValueError("replica_task.admission must be a dict")
    _reject_unknown_fields(
        raw_admission,
        {
            "policy",
            "score_threshold",
            "score_bypass_threshold",
            "admission_ratio",
            "min_replica_pages",
            "max_replica_pages_per_batch",
        },
        "replica_task.admission",
    )

    policy = str(raw_admission.get("policy", "eager_all")).strip()
    if policy not in _REPLICA_TASK_POLICIES:
        raise ValueError(
            "replica_task.admission.policy must be one of "
            f"{sorted(_REPLICA_TASK_POLICIES)}, got {policy!r}"
        )

    admission = _FluxonReplicaTaskAdmissionConfig(
        policy=policy,
        score_threshold=_as_float(
            raw_admission.get("score_threshold", 0.55),
            "replica_task.admission.score_threshold",
        ),
        score_bypass_threshold=_as_float(
            raw_admission.get("score_bypass_threshold", 0.82),
            "replica_task.admission.score_bypass_threshold",
        ),
        admission_ratio=_as_float(
            raw_admission.get("admission_ratio", 1.0),
            "replica_task.admission.admission_ratio",
        ),
        min_replica_pages=_as_non_negative_int(
            raw_admission.get("min_replica_pages", 0),
            "replica_task.admission.min_replica_pages",
        ),
        max_replica_pages_per_batch=_as_non_negative_int(
            raw_admission.get("max_replica_pages_per_batch", STORAGE_BATCH_SIZE),
            "replica_task.admission.max_replica_pages_per_batch",
        ),
    )

    return _FluxonReplicaTaskConfig(
        enabled=_as_bool(raw_config.get("enabled", True), "replica_task.enabled"),
        admission=admission,
        metrics_sample_interval_ms=max(
            0,
            _as_non_negative_int(
                raw_config.get("metrics_sample_interval_ms", 1000),
                "replica_task.metrics_sample_interval_ms",
            ),
        ),
    )


class HiCacheFluxon(HiCacheStorage):
    """Fluxon-backed HiCache storage with v1 and hostless data paths."""

    def __init__(self, storage_config: HiCacheStorageConfig):
        self.storage_config = storage_config
        self.extra_config = dict(storage_config.extra_config or {})
        unknown_config = sorted(set(self.extra_config) - {"config_path"})
        if unknown_config:
            raise ValueError(
                "Fluxon HiCache extra_config only accepts 'config_path'; "
                f"unknown field(s): {unknown_config}"
            )
        config_path = self.extra_config.get("config_path")
        if not config_path:
            raise RuntimeError(
                "Fluxon HiCache backend requires extra_config['config_path']."
            )
        self.extra_backend_tag = None
        self.key_prefix = ""
        self.enable_storage_metrics = storage_config.enable_storage_metrics
        self.prefetch_pgs: list[int] = []
        self.backup_pgs: list[int] = []
        self.prefetch_bandwidth: list[float] = []
        self.backup_bandwidth: list[float] = []
        self._last_observability_snapshot: dict[str, Any] | None = None
        self._warm_futures: dict[str, Any] = {}
        self._warm_inflight: set[str] = set()
        self._warm_lock = threading.Lock()
        self._enable_warm_get = True
        self._warm_limit = _E44_WARM_PENDING_LIMIT
        self._warm_submit_limit = _E44_WARM_SUBMIT_LIMIT
        self._warm_drain_budget = _E44_WARM_DRAIN_BUDGET
        self._batch_concurrency = _E44_BATCH_CONCURRENCY
        self._batch_exists_pin_ttl_ms = _E44_BATCH_EXISTS_PIN_TTL_MS
        self._warm_batch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fluxon-warm",
        )
        self._put_opts: Any | None = None
        self._put_opts_make_replica: Any | None = None
        self._put_opts_local_only: Any | None = None
        self._put_optional_args_type: Any | None = None
        self._put_write_through = False
        self._local_fast_put_start_direct_client: Any | None = None
        self._local_fast_put_start_direct_supported_kwargs: set[str] | None = None
        self._local_fast_put_start_direct_accepts_var_keyword = False
        self._local_fast_put_start_use_direct = False
        self._local_fast_put_start_direct_warned = False
        self._replica_task_config = _parse_replica_task_config(
            {
                "replica_task": {
                    "enabled": True,
                    "admission": {
                        "policy": "prefix_end_depth_ratio",
                        "admission_ratio": 1.0,
                        "min_replica_pages": 8,
                        "max_replica_pages_per_batch": 288,
                    },
                    "metrics_sample_interval_ms": 1_000,
                }
            }
        )
        self._replica_task_candidate_pages = 0
        self._replica_task_admitted_pages = 0
        self._replica_task_skipped_pages = 0
        self._replica_task_last_log_ts = 0.0
        self._fluxon_cuda_segment_registration_done = False
        self._gpu_direct_staging_pool: Optional[_FluxonGpuStagingPool] = None
        logger.info(
            "Fluxon batch_exists pin_ttl_ms=%s",
            self._batch_exists_pin_ttl_ms,
        )
        self._fluxon_cuda_segment_registration_required = False
        self._init_key_namespace(storage_config)

        (
            FluxonKvClientConfig,
            new_store,
            PutOptionalArgs,
            FixedSlabAllocator,
        ) = _import_fluxon_symbols()
        self._fixed_slab_allocator_type = FixedSlabAllocator
        self._put_optional_args_type = PutOptionalArgs
        hicache_write_policy = "write_back"
        update_master_router = False
        self._put_write_through = update_master_router
        self._put_opts_make_replica = _build_put_optional_args(
            PutOptionalArgs,
            reject_if_inflight_same_key=True,
            reject_if_exist_same_key=True,
            write_through=update_master_router,
            make_replica_task=True,
        )
        self._put_opts_local_only = _build_put_optional_args(
            PutOptionalArgs,
            reject_if_inflight_same_key=True,
            reject_if_exist_same_key=True,
            write_through=update_master_router,
            make_replica_task=False,
        )
        self._put_opts = self._put_opts_make_replica
        logger.info(
            "Fluxon HiCache put options: hicache_write_policy=%s "
            "update_master_router=%s legacy_write_through=%s "
            "replica_task_enabled=%s replica_task_policy=%s score_threshold=%.3f "
            "admission_ratio=%.3f max_replica_pages_per_batch=%d",
            hicache_write_policy or "<unset>",
            update_master_router,
            update_master_router,
            self._replica_task_config.enabled,
            self._replica_task_config.admission.policy,
            self._replica_task_config.admission.score_threshold,
            self._replica_task_config.admission.admission_ratio,
            self._replica_task_config.admission.max_replica_pages_per_batch,
        )
        config = FluxonKvClientConfig.from_file(config_path)
        self.key_prefix = str(config.instance_key).rstrip(":")
        config_dict = config.to_dict()
        zero_contribution = _is_zero_contribution_config(config_dict)
        self._fluxon_cuda_segment_registration_required = (
            torch.cuda.is_available() and zero_contribution
        )
        config = FluxonKvClientConfig(
            _namespace_config_for_process(config, config_dict)
        )
        logger.info(
            "Fluxon HiCache CUDA segment registration: required=%s zero_contribution=%s cuda_available=%s",
            self._fluxon_cuda_segment_registration_required,
            zero_contribution,
            torch.cuda.is_available(),
        )

        store_result = new_store(config)
        if not store_result.is_ok():
            raise RuntimeError(
                f"Failed to initialize Fluxon HiCache store: "
                f"{store_result.unwrap_error()}"
            )
        self.store = store_result.unwrap()
        self._init_local_fast_put_start_capabilities()
        self._batch_is_exist_accepts_pin_ttl = self._detect_batch_is_exist_pin_ttl_support()
        if (
            not self._batch_is_exist_accepts_pin_ttl
            and self._batch_exists_pin_ttl_ms is not None
        ):
            logger.info(
                "Fluxon batch_is_exist does not accept pin_ttl; disabling SGLang batch_exists pin TTL path"
            )
            self._batch_exists_pin_ttl_ms = None

    def _init_local_fast_put_start_capabilities(self) -> None:
        direct_client = getattr(self.store, "_client", None)
        direct_method = getattr(direct_client, "local_fast_put_start", None)
        self._local_fast_put_start_direct_client = direct_client
        if direct_method is None:
            logger.info(
                "Fluxon local_fast_put_start capability: direct PyO3 client unavailable"
            )
            return

        supported = _method_supported_kwargs(direct_method)
        if supported is None:
            logger.info(
                "Fluxon local_fast_put_start capability: direct PyO3 signature unavailable"
            )
            return

        supported_kwargs, accepts_var_keyword = supported
        self._local_fast_put_start_direct_supported_kwargs = supported_kwargs
        self._local_fast_put_start_direct_accepts_var_keyword = accepts_var_keyword
        self._local_fast_put_start_use_direct = (
            not accepts_var_keyword and "write_through" not in supported_kwargs
        )
        logger.info(
            "Fluxon local_fast_put_start capability: direct_supported_kwargs=%s "
            "accepts_var_keyword=%s use_direct=%s",
            sorted(supported_kwargs),
            accepts_var_keyword,
            self._local_fast_put_start_use_direct,
        )

    def _detect_batch_is_exist_pin_ttl_support(self) -> bool:
        batch_is_exist = getattr(self.store, "batch_is_exist", None)
        if batch_is_exist is None:
            return False
        try:
            signature = inspect.signature(batch_is_exist)
        except (TypeError, ValueError):
            return True
        return "pin_ttl" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _ensure_fluxon_local_segments_cuda_registered(self) -> None:
        if self._fluxon_cuda_segment_registration_done:
            return
        if not self._fluxon_cuda_segment_registration_required:
            return

        wait_ready = getattr(self.store, "wait_local_segments_ready", None)
        if wait_ready is None:
            raise RuntimeError(
                "Fluxon store does not expose wait_local_segments_ready() required for direct H2D restore"
            )
        segment_list = wait_ready()
        if not isinstance(segment_list, list) or len(segment_list) == 0:
            raise RuntimeError(
                "wait_local_segments_ready() must return a non-empty segment list"
            )

        cudart = torch.cuda.cudart()
        registered_now = 0
        external_segments = 0
        for segment in segment_list:
            if not isinstance(segment, dict):
                raise RuntimeError(
                    "wait_local_segments_ready() segment entries must be dicts"
                )
            segment_label = str(segment.get("segment_label", ""))
            if not segment_label.startswith("external_owner:"):
                continue
            external_segments += 1
            segment_len = int(segment["len"])
            generation = int(segment["generation"])
            node_id = str(segment.get("node_id", ""))
            if segment_len <= 0:
                raise RuntimeError(
                    f"Fluxon local segment len must be > 0, got {segment_len}"
                )
            ptr_fields = [("write_ptr", False)]
            if "read_ptr" in segment:
                ptr_fields.append(("read_ptr", True))
            for ptr_field, read_only in ptr_fields:
                register_ptr = int(segment[ptr_field])
                if register_ptr <= 0:
                    raise RuntimeError(
                        f"Fluxon local segment {ptr_field} must be > 0, got {register_ptr}"
                    )
                segment_key = (register_ptr, segment_len, generation)
                with _REGISTERED_FLUXON_SEGMENTS_LOCK:
                    if segment_key in _REGISTERED_FLUXON_SEGMENTS:
                        continue
                    register_flags = (
                        _CUDA_HOST_REGISTER_PORTABLE | _CUDA_HOST_REGISTER_MAPPED
                    )
                    if read_only:
                        register_flags |= _CUDA_HOST_REGISTER_READ_ONLY
                    rc = cudart.cudaHostRegister(
                        register_ptr,
                        segment_len,
                        register_flags,
                    )
                    if int(rc) != 0 and read_only:
                        logger.warning(
                            "cudaHostRegister with ReadOnly failed (rc=%d, %s) for Fluxon "
                            "segment %s=%#x len=%d generation=%d; retrying without ReadOnly",
                            int(rc),
                            cudart.cudaGetErrorString(rc),
                            ptr_field,
                            register_ptr,
                            segment_len,
                            generation,
                        )
                        register_flags = (
                            _CUDA_HOST_REGISTER_PORTABLE | _CUDA_HOST_REGISTER_MAPPED
                        )
                        rc = cudart.cudaHostRegister(
                            register_ptr,
                            segment_len,
                            register_flags,
                        )
                    if int(rc) != 0:
                        raise RuntimeError(
                            f"cudaHostRegister failed (rc={int(rc)}, {cudart.cudaGetErrorString(rc)}) "
                            f"for Fluxon segment {ptr_field}={register_ptr:#x} len={segment_len} "
                            f"generation={generation} node_id={node_id} flags={register_flags}"
                        )
                    _REGISTERED_FLUXON_SEGMENTS.add(segment_key)
                    registered_now += 1

        if external_segments == 0:
            raise RuntimeError(
                "Fluxon external CUDA host register requires an external_owner segment, "
                f"but wait_local_segments_ready() returned {len(segment_list)} segment(s)."
            )
        self._fluxon_cuda_segment_registration_done = True
        logger.info(
            "Fluxon external CUDA host register complete: external_segments=%d newly_registered=%d ptr_fields=write_ptr/read_ptr",
            external_segments,
            registered_now,
        )

    def _assert_fluxon_cuda_segments_registered(self, api_name: str) -> None:
        if (
            self._fluxon_cuda_segment_registration_required
            and not self._fluxon_cuda_segment_registration_done
        ):
            raise RuntimeError(
                f"Fluxon {api_name} requires CUDA segment registration before use. "
                "register_mem_pool_host/register_mem_host_pool_v2 must run during storage lifecycle."
            )

    def _init_key_namespace(self, storage_config: HiCacheStorageConfig) -> None:
        tp_rank = storage_config.tp_rank
        tp_size = storage_config.tp_size
        pp_rank = storage_config.pp_rank
        pp_size = storage_config.pp_size
        is_mla_model = storage_config.is_mla_model
        model_name = storage_config.model_name or ""
        model_name = "-".join(model_name.split("/")) if model_name else ""

        suffix = f"_{model_name}" if model_name else ""
        replica_admission_suffix = suffix
        if not is_mla_model:
            suffix += f"_{tp_rank}_{tp_size}"
            replica_admission_suffix += f"_tp{tp_size}"
        if pp_size > 1:
            suffix += f"_{pp_size}_{pp_rank}"
            replica_admission_suffix += f"_{pp_size}_{pp_rank}"
        self.config_suffix = suffix
        self._replica_admission_config_suffix = replica_admission_suffix

    def _component_suffix(self, component_name: Optional[Any] = None) -> Optional[str]:
        if component_name is None or component_name in ("__default__", PoolName.KV):
            return None
        if isinstance(component_name, PoolName):
            return component_name.value
        return str(component_name)

    def _logical_component_key(
        self, key: str, component_name: Optional[Any] = None
    ) -> str:
        component = self._component_suffix(component_name)
        logical_key = key if component is None else f"{key}.{component}"
        logical_key += self.config_suffix
        if self.extra_backend_tag:
            logical_key = f"{self.extra_backend_tag}_{logical_key}"
        return logical_key

    def _store_key(self, key: str, component_name: Optional[Any] = None) -> str:
        logical_key = self._logical_component_key(key, component_name)
        if not self.key_prefix:
            return logical_key
        return f"{self.key_prefix}:{logical_key}"

    def register_mem_pool_host(self, mem_pool_host):
        super().register_mem_pool_host(mem_pool_host)
        self._ensure_fluxon_local_segments_cuda_registered()

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        self._ensure_fluxon_local_segments_cuda_registered()

    def _batch_exists_flags(
        self,
        storage_keys: List[str],
        *,
        pin_ttl_ms: Optional[int] = None,
    ) -> List[bool]:
        if not storage_keys:
            return []
        if hasattr(self.store, "batch_is_exist"):
            start = time.perf_counter()
            if pin_ttl_ms is not None and getattr(
                self, "_batch_is_exist_accepts_pin_ttl", False
            ):
                raw_results = self.store.batch_is_exist(
                    storage_keys, pin_ttl=pin_ttl_ms
                )
            else:
                raw_results = self.store.batch_is_exist(storage_keys)
            results = [result == 1 for result in raw_results]
            logger.info(
                "Fluxon batch_is_exist: keys=%d hits=%d misses=%d pin_ttl_ms=%s duration_ms=%.3f",
                len(storage_keys),
                sum(1 for result in results if result),
                sum(1 for result in results if not result),
                str(pin_ttl_ms),
                (time.perf_counter() - start) * 1000.0,
            )
            return results

        start = time.perf_counter()
        results: list[bool] = []
        for storage_key in storage_keys:
            exists_result = self.store.is_exist(storage_key)
            if exists_result.is_ok():
                results.append(bool(exists_result.unwrap()))
            else:
                _ = exists_result.unwrap_error()
                results.append(False)
        logger.info(
            "Fluxon sequential_is_exist: keys=%d hits=%d misses=%d duration_ms=%.3f",
            len(storage_keys),
            sum(1 for result in results if result),
            sum(1 for result in results if not result),
            (time.perf_counter() - start) * 1000.0,
        )
        return results

    def _prefix_hit_pages(self, exists_flags: List[bool]) -> int:
        hit_pages = 0
        for exists in exists_flags:
            if not exists:
                break
            hit_pages += 1
        return hit_pages

    def _pin_existing_prefix(
        self,
        storage_keys: List[str],
        hit_pages: int,
        *,
        reason: str,
    ) -> int:
        if hit_pages <= 0:
            return 0
        pin_ttl_ms = self._batch_exists_pin_ttl_ms
        if pin_ttl_ms is None:
            return hit_pages
        if not hasattr(self.store, "batch_is_exist"):
            logger.warning(
                "Fluxon batch_exists pin skipped: store has no batch_is_exist reason=%s",
                reason,
            )
            return hit_pages

        candidate = min(hit_pages, len(storage_keys))
        while candidate > 0:
            pin_flags = self._batch_exists_flags(
                storage_keys[:candidate], pin_ttl_ms=pin_ttl_ms
            )
            if len(pin_flags) == candidate and all(pin_flags):
                if candidate != hit_pages:
                    logger.warning(
                        "Fluxon batch_exists pin shortened prefix: reason=%s original=%d pinned=%d",
                        reason,
                        hit_pages,
                        candidate,
                    )
                return candidate
            candidate = self._prefix_hit_pages(pin_flags)

        logger.warning(
            "Fluxon batch_exists pin failed for prefix: reason=%s original=%d",
            reason,
            hit_pages,
        )
        return 0

    def _batch_exists_v2_storage_keys(
        self,
        kv_storage_keys: List[str],
        extra_pool_keys: dict[Any, List[str]],
        pool_transfers: Optional[List[PoolTransfer]],
        final_pages: int,
    ) -> List[str]:
        storage_keys = list(kv_storage_keys[:final_pages])
        for transfer in pool_transfers or []:
            component_keys = extra_pool_keys.get(transfer.name)
            if not component_keys or final_pages == 0:
                continue
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                storage_keys.extend(component_keys[:final_pages])
            else:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                storage_keys.extend(
                    component_keys[max(0, final_pages - trailing) : final_pages]
                )
        return storage_keys

    def _pin_batch_exists_v2_prefix(
        self,
        kv_storage_keys: List[str],
        extra_pool_keys: dict[Any, List[str]],
        pool_transfers: Optional[List[PoolTransfer]],
        final_pages: int,
    ) -> int:
        if final_pages <= 0:
            return 0
        pin_ttl_ms = self._batch_exists_pin_ttl_ms
        if pin_ttl_ms is None:
            return final_pages
        if not hasattr(self.store, "batch_is_exist"):
            logger.warning(
                "Fluxon batch_exists_v2 pin skipped: store has no batch_is_exist"
            )
            return final_pages

        candidate = final_pages
        while candidate > 0:
            storage_keys = self._batch_exists_v2_storage_keys(
                kv_storage_keys, extra_pool_keys, pool_transfers, candidate
            )
            pin_flags = self._batch_exists_flags(storage_keys, pin_ttl_ms=pin_ttl_ms)
            if len(pin_flags) == len(storage_keys) and all(pin_flags):
                if candidate != final_pages:
                    logger.warning(
                        "Fluxon batch_exists_v2 pin shortened prefix: original=%d pinned=%d",
                        final_pages,
                        candidate,
                    )
                return candidate
            candidate -= 1

        logger.warning(
            "Fluxon batch_exists_v2 pin failed for prefix: original=%d",
            final_pages,
        )
        return 0

    def batch_missing_indices(
        self,
        keys: List[str],
        component_name: Optional[Any] = None,
    ) -> List[int]:
        if not keys:
            return []
        storage_keys = [self._store_key(key, component_name) for key in keys]
        exists_flags = self._batch_exists_flags(storage_keys)
        missing_indices = [
            index for index, exists in enumerate(exists_flags) if not exists
        ]
        logger.info(
            "Fluxon batch_missing_indices before local_fast_put_start: keys=%d existing=%d missing=%d component=%s",
            len(keys),
            len(keys) - len(missing_indices),
            len(missing_indices),
            component_name,
        )
        return missing_indices

    def _record_prefetch_metrics(self, total_pages: int, total_bytes: int, start: float):
        if not self.enable_storage_metrics or total_pages <= 0:
            return
        elapsed = max(time.perf_counter() - start, 1e-9)
        self.prefetch_pgs.append(total_pages)
        self.prefetch_bandwidth.append(total_bytes / (1024**3) / elapsed)

    def _record_backup_metrics(self, total_pages: int, total_bytes: int, start: float):
        if not self.enable_storage_metrics or total_pages <= 0:
            return
        elapsed = max(time.perf_counter() - start, 1e-9)
        self.backup_pgs.append(total_pages)
        self.backup_bandwidth.append(total_bytes / (1024**3) / elapsed)

    def _put_opts_for_replica_task(self, make_replica_task: bool) -> Any:
        return (
            self._put_opts_make_replica
            if make_replica_task
            else self._put_opts_local_only
        )

    def _put_opts_for_replica_task_mask(
        self,
        admission_mask: List[bool],
        radix_parent_keys: Optional[List[Optional[str]]],
        content_depths: Optional[List[int]],
        atomic_group_lens: List[int],
    ) -> Any:
        put_optional_args_type = self._put_optional_args_type
        assert put_optional_args_type is not None
        return _build_put_optional_args(
            put_optional_args_type,
            reject_if_inflight_same_key=True,
            reject_if_exist_same_key=True,
            write_through=self._put_write_through,
            make_replica_task=any(admission_mask),
            make_replica_task_mask=[bool(item) for item in admission_mask],
            radix_parent_keys=(
                None if radix_parent_keys is None else list(radix_parent_keys)
            ),
            content_depths=(None if content_depths is None else list(content_depths)),
            atomic_group_lens=list(atomic_group_lens),
        )

    def _local_fast_put_start_direct_kwarg_supported(self, name: str) -> bool:
        if self._local_fast_put_start_direct_accepts_var_keyword:
            return True
        supported = self._local_fast_put_start_direct_supported_kwargs
        if supported is None:
            return name in (
                "reject_if_inflight_same_key",
                "reject_if_exist_same_key",
            )
        return name in supported

    def _call_direct_local_fast_put_start(
        self,
        storage_keys: List[str],
        value_len: int,
        make_replica_task: bool,
        admission_mask: Optional[List[bool]] = None,
        radix_parent_keys: Optional[List[Optional[str]]] = None,
        content_depths: Optional[List[int]] = None,
        atomic_group_lens: Optional[List[int]] = None,
    ) -> int:
        direct_client = self._local_fast_put_start_direct_client
        direct_method = getattr(direct_client, "local_fast_put_start", None)
        if direct_method is None:
            raise RuntimeError(
                "Fluxon direct local_fast_put_start fallback requires store._client"
            )

        kwargs: dict[str, Any] = {}
        if self._local_fast_put_start_direct_kwarg_supported(
            "reject_if_inflight_same_key"
        ):
            kwargs["reject_if_inflight_same_key"] = True
        if self._local_fast_put_start_direct_kwarg_supported(
            "reject_if_exist_same_key"
        ):
            kwargs["reject_if_exist_same_key"] = True
        if self._local_fast_put_start_direct_kwarg_supported("write_through"):
            kwargs["write_through"] = self._put_write_through
        if self._local_fast_put_start_direct_kwarg_supported("make_replica_task"):
            kwargs["make_replica_task"] = make_replica_task
        if (
            admission_mask is not None
            and self._local_fast_put_start_direct_kwarg_supported(
                "make_replica_task_mask"
            )
        ):
            kwargs["make_replica_task_mask"] = [bool(item) for item in admission_mask]
        if (
            radix_parent_keys is not None
            and self._local_fast_put_start_direct_kwarg_supported(
                "radix_parent_keys"
            )
        ):
            kwargs["radix_parent_keys"] = list(radix_parent_keys)
        if (
            content_depths is not None
            and self._local_fast_put_start_direct_kwarg_supported("content_depths")
        ):
            kwargs["content_depths"] = list(content_depths)
        if (
            atomic_group_lens is not None
            and self._local_fast_put_start_direct_kwarg_supported(
                "atomic_group_lens"
            )
        ):
            kwargs["atomic_group_lens"] = list(atomic_group_lens)

        if not self._local_fast_put_start_direct_warned:
            omitted = []
            for name in (
                "write_through",
                "make_replica_task",
                "make_replica_task_mask",
                "radix_parent_keys",
                "content_depths",
                "atomic_group_lens",
            ):
                if not self._local_fast_put_start_direct_kwarg_supported(name):
                    omitted.append(name)
            if omitted:
                logger.warning(
                    "Fluxon local_fast_put_start using direct PyO3 compatibility "
                    "path; omitted unsupported kwargs=%s put_write_through=%s "
                    "make_replica_task=%s",
                    omitted,
                    self._put_write_through,
                    make_replica_task,
                )
            self._local_fast_put_start_direct_warned = True

        result = direct_method(storage_keys, value_len, **kwargs)
        if hasattr(result, "is_ok"):
            if not result.is_ok():
                err = result.unwrap_error()
                if isinstance(err, Exception):
                    raise err
                raise RuntimeError(f"local_fast_put_start backend error: {err}")
            result = result.unwrap()
        if not isinstance(result, int) or result <= 0:
            raise RuntimeError(f"local_fast_put_start returned invalid plan_ptr: {result!r}")
        return int(result)

    def _call_local_fast_put_start(
        self,
        storage_keys: List[str],
        value_len: int,
        opts: Any,
        make_replica_task: bool,
        admission_mask: Optional[List[bool]] = None,
        radix_parent_keys: Optional[List[Optional[str]]] = None,
        content_depths: Optional[List[int]] = None,
        atomic_group_lens: Optional[List[int]] = None,
    ) -> int:
        if self._local_fast_put_start_use_direct:
            return self._call_direct_local_fast_put_start(
                storage_keys,
                value_len,
                make_replica_task=make_replica_task,
                admission_mask=admission_mask,
                radix_parent_keys=radix_parent_keys,
                content_depths=content_depths,
                atomic_group_lens=atomic_group_lens,
            )

        try:
            return int(
                self.store.local_fast_put_start(
                    storage_keys,
                    value_len,
                    opts=opts,
                )
            )
        except TypeError as exc:
            if not _is_unexpected_fluxon_put_kwarg(exc):
                raise
            self._local_fast_put_start_use_direct = True
            logger.warning(
                "Fluxon local_fast_put_start wrapper rejected a replica option; "
                "switching this process to direct PyO3 compatibility path: %s",
                exc,
            )
            return self._call_direct_local_fast_put_start(
                storage_keys,
                value_len,
                make_replica_task=make_replica_task,
                admission_mask=admission_mask,
                radix_parent_keys=radix_parent_keys,
                content_depths=content_depths,
                atomic_group_lens=atomic_group_lens,
            )

    @staticmethod
    def _stable_fraction(value: str) -> float:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") / float(1 << 64)

    def _replica_admission_key(self, storage_key: str) -> str:
        physical_suffix = self.config_suffix
        if not physical_suffix:
            return storage_key
        if not storage_key.endswith(physical_suffix):
            raise ValueError(
                "Fluxon replica admission key does not end in the configured "
                f"physical suffix: key={storage_key!r} suffix={physical_suffix!r}"
            )
        return (
            storage_key[: -len(physical_suffix)]
            + self._replica_admission_config_suffix
        )

    @staticmethod
    def _extra_info_dict(extra_info: Optional[HiCacheStorageExtraInfo]) -> dict[str, Any]:
        if extra_info is None or extra_info.extra_info is None:
            return {}
        if not isinstance(extra_info.extra_info, dict):
            raise ValueError("HiCacheStorageExtraInfo.extra_info must be a dict")
        return extra_info.extra_info

    @staticmethod
    def _score_map_from_extra(extra_payload: dict[str, Any]) -> dict[str, float]:
        for key in (
            "replica_task_scores",
            "kv_scores",
            "page_scores",
            "scores",
        ):
            raw_scores = extra_payload.get(key)
            if raw_scores is None:
                continue
            if not isinstance(raw_scores, dict):
                raise ValueError(f"extra_info.extra_info[{key!r}] must be a dict")
            return {str(k): float(v) for k, v in raw_scores.items()}
        return {}

    def _replica_task_score(
        self,
        storage_key: str,
        page_index: int,
        page_count: int,
        extra_info: Optional[HiCacheStorageExtraInfo],
        score_map: dict[str, float],
    ) -> float:
        if storage_key in score_map:
            return max(0.0, min(1.0, score_map[storage_key]))

        logical_key = storage_key
        if self.key_prefix and storage_key.startswith(f"{self.key_prefix}:"):
            logical_key = storage_key[len(self.key_prefix) + 1 :]
        if logical_key in score_map:
            return max(0.0, min(1.0, score_map[logical_key]))

        prefix_bonus = 0.0
        if extra_info is not None and extra_info.prefix_keys:
            prefix_bonus = min(len(extra_info.prefix_keys), 64) / 640.0

        if page_count <= 1:
            position_score = 0.5
        else:
            position_score = 1.0 - (page_index / float(page_count - 1))
        jitter = self._stable_fraction(
            self._replica_admission_key(storage_key)
        )
        score = (0.45 * position_score) + (0.45 * jitter) + prefix_bonus
        return max(0.0, min(1.0, score))

    def _record_replica_task_admission(
        self,
        candidate_pages: int,
        admitted_pages: int,
        op_name: str,
    ) -> None:
        skipped_pages = candidate_pages - admitted_pages
        self._replica_task_candidate_pages += candidate_pages
        self._replica_task_admitted_pages += admitted_pages
        self._replica_task_skipped_pages += skipped_pages

        interval_ms = self._replica_task_config.metrics_sample_interval_ms
        now = time.perf_counter()
        if interval_ms == 0 or (
            now - self._replica_task_last_log_ts
        ) * 1000.0 >= interval_ms:
            self._replica_task_last_log_ts = now
            logger.info(
                "Fluxon replica task admission: op=%s policy=%s candidate=%d "
                "admitted=%d skipped=%d total_candidate=%d total_admitted=%d "
                "total_skipped=%d",
                op_name,
                self._replica_task_config.admission.policy,
                candidate_pages,
                admitted_pages,
                skipped_pages,
                self._replica_task_candidate_pages,
                self._replica_task_admitted_pages,
                self._replica_task_skipped_pages,
            )

    @staticmethod
    def _normalize_atomic_group_lens(
        storage_keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo],
    ) -> List[int]:
        page_count = len(storage_keys)
        raw_group_lens = (
            None if extra_info is None else extra_info.atomic_group_lens
        )
        if raw_group_lens is None:
            return [1] * page_count
        if not isinstance(raw_group_lens, list):
            raise ValueError("HiCacheStorageExtraInfo.atomic_group_lens must be list[int]")
        group_lens: list[int] = []
        for index, group_len in enumerate(raw_group_lens):
            if type(group_len) is not int:
                raise ValueError(
                    "HiCacheStorageExtraInfo.atomic_group_lens items must be int: "
                    f"index={index} got={type(group_len)}"
                )
            if group_len <= 0:
                raise ValueError(
                    "HiCacheStorageExtraInfo.atomic_group_lens entries must be > 0: "
                    f"index={index} got={group_len}"
                )
            group_lens.append(group_len)
        group_sum = sum(group_lens)
        if group_sum != page_count:
            raise ValueError(
                "HiCacheStorageExtraInfo.atomic_group_lens must sum to key count: "
                f"sum={group_sum} keys={page_count}"
            )
        return group_lens

    @staticmethod
    def _atomic_group_ranges(group_lens: List[int]) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        offset = 0
        for group_len in group_lens:
            ranges.append((offset, offset + group_len))
            offset += group_len
        return ranges

    @staticmethod
    def _absolute_content_depths(
        storage_keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo],
    ) -> Optional[List[int]]:
        prefix_hashes = None if extra_info is None else extra_info.prefix_keys
        if prefix_hashes is None:
            return None
        prefix_pages = len(prefix_hashes)
        max_depth = prefix_pages + len(storage_keys) - 1
        if max_depth > 0xFFFFFFFF:
            raise ValueError(
                "Fluxon content depth exceeds u32: "
                f"prefix_pages={prefix_pages} keys={len(storage_keys)}"
            )
        return [prefix_pages + index for index in range(len(storage_keys))]

    def _radix_parent_keys(
        self,
        storage_keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo],
        component_name: Optional[Any] = None,
    ) -> Optional[List[Optional[str]]]:
        prefix_hashes = None if extra_info is None else extra_info.prefix_keys
        if prefix_hashes is None:
            return None
        if not storage_keys:
            return []
        first_parent = (
            None
            if not prefix_hashes
            else self._store_key(prefix_hashes[-1], component_name)
        )
        return [first_parent, *storage_keys[:-1]]

    def _replica_task_admission_mask(
        self,
        storage_keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo],
        op_name: str,
        record_metrics: bool = True,
    ) -> List[bool]:
        page_count = len(storage_keys)
        if page_count == 0:
            return []
        group_lens = self._normalize_atomic_group_lens(storage_keys, extra_info)
        group_ranges = self._atomic_group_ranges(group_lens)

        cfg = self._replica_task_config
        admission = cfg.admission
        if not cfg.enabled:
            mask = [False] * page_count
            if record_metrics:
                self._record_replica_task_admission(page_count, 0, op_name)
            return mask

        if admission.policy == "eager_all":
            mask = [True] * page_count
            if record_metrics:
                self._record_replica_task_admission(page_count, page_count, op_name)
            return mask

        # The first radix node contains the long, group-shared system prefix.  It is
        # both the highest-value part of this workload and the part for which a
        # single missing page truncates every later storage hit to zero.  Keep this
        # policy Fluxon-specific: duplicate only that complete atomic node on the
        # remote CPU and let owner-hot exclusive demotion handle all descendants.
        # `prefix_keys` is empty only for a direct child of the radix root; `None`
        # means the caller did not provide dependency metadata and must fail closed.
        if admission.policy == "prefix_root_only":
            min_root_pages = admission.min_replica_pages
            max_root_pages = admission.max_replica_pages_per_batch
            admit_root = (
                extra_info is not None
                and extra_info.prefix_keys == []
                and page_count >= min_root_pages
                and (max_root_pages == 0 or page_count <= max_root_pages)
            )
            mask = [admit_root] * page_count
            if record_metrics:
                self._record_replica_task_admission(
                    page_count,
                    page_count if admit_root else 0,
                    op_name,
                )
            return mask

        # A physical radix-root test is insertion-order dependent: after the
        # common prefix splits, later per-session system-prefix nodes are no
        # longer direct children of the root. Select complete atomic nodes by
        # logical prefix depth instead. Nodes starting before the configured
        # depth limit are kept whole, including a node crossing the boundary.
        # The anchor hash makes ratio decisions stable across TP ranks; nodes
        # before the anchor are shared ancestors and are always admitted.
        if admission.policy in ("prefix_depth_ratio", "prefix_end_depth_ratio"):
            prefix_hashes = None if extra_info is None else extra_info.prefix_keys
            prefix_pages = 0 if prefix_hashes is None else len(prefix_hashes)
            depth_limit = admission.max_replica_pages_per_batch
            admit_branch = False
            if prefix_hashes is not None:
                anchor_pages = max(1, admission.min_replica_pages)
                path_storage_keys = [
                    self._store_key(prefix_hash) for prefix_hash in prefix_hashes
                ] + list(storage_keys)
                if len(path_storage_keys) < anchor_pages:
                    admit_branch = True
                else:
                    anchor_key = path_storage_keys[anchor_pages - 1]
                    admit_branch = (
                        self._stable_fraction(
                            self._replica_admission_key(anchor_key)
                        )
                        < admission.admission_ratio
                    )

            # `prefix_depth_ratio` admits a complete atomic node when its start
            # is before the limit.  That is intentionally permissive, but a
            # newly routed request can be represented as one root child whose
            # key contains the *entire* prompt.  Admitting such a node on every
            # later turn lets a nominal depth-160 policy copy 300+ page prompts
            # and continuously churn the remote CPU cache.
            #
            # `prefix_end_depth_ratio` is the bounded variant: decide each
            # atomic group independently and admit it only when the complete
            # group ends at or before the logical depth limit.  It never splits
            # an atomic group.  With depth 288 this keeps the observed turn-0
            # system prefixes while rejecting progressively longer root nodes.
            mask: list[bool] = []
            for start, end in group_ranges:
                if admission.policy == "prefix_end_depth_ratio":
                    within_depth = prefix_hashes is not None and (
                        depth_limit == 0 or prefix_pages + end <= depth_limit
                    )
                else:
                    within_depth = prefix_hashes is not None and (
                        depth_limit == 0 or prefix_pages + start < depth_limit
                    )
                mask.extend([admit_branch and within_depth] * (end - start))
            if record_metrics:
                self._record_replica_task_admission(
                    page_count,
                    sum(1 for item in mask if item),
                    op_name,
                )
            return mask

        scores: list[float] | None = None
        if admission.policy == "ratio_only":
            page_candidates = [
                self._stable_fraction(self._replica_admission_key(storage_key))
                < admission.admission_ratio
                for storage_key in storage_keys
            ]
        else:
            extra_payload = self._extra_info_dict(extra_info)
            score_map = self._score_map_from_extra(extra_payload)
            scores = [
                self._replica_task_score(
                    storage_key,
                    index,
                    page_count,
                    extra_info,
                    score_map,
                )
                for index, storage_key in enumerate(storage_keys)
            ]
            if admission.policy == "kv_score_only":
                page_candidates = [
                    score >= admission.score_threshold for score in scores
                ]
            elif admission.policy == "kv_score_low_only":
                page_candidates = [
                    score <= admission.score_threshold for score in scores
                ]
            elif admission.policy == "kv_score_pressure_budget":
                page_candidates = [
                    score >= admission.score_bypass_threshold
                    or (
                        score >= admission.score_threshold
                        and self._stable_fraction(
                            self._replica_admission_key(storage_keys[index])
                        )
                        < admission.admission_ratio
                    )
                    for index, score in enumerate(scores)
                ]
            else:
                raise ValueError(
                    f"unsupported replica task admission policy: {admission.policy}"
                )

        if scores is None:
            scores = [
                self._replica_task_score(
                    storage_key,
                    index,
                    page_count,
                    extra_info,
                    {},
                )
                for index, storage_key in enumerate(storage_keys)
            ]

        group_decisions: list[bool] = []
        group_priorities: list[float] = []
        for group_index, (start, end) in enumerate(group_ranges):
            group_len = end - start
            candidate_probability = (
                admission.admission_ratio
                if admission.policy == "ratio_only"
                else sum(1 for item in page_candidates[start:end] if item)
                / float(group_len)
            )
            group_identity = (
                f"atomic-group:{group_len}:"
                f"{self._replica_admission_key(storage_keys[start])}:"
                f"{self._replica_admission_key(storage_keys[end - 1])}"
            )
            group_decisions.append(
                self._stable_fraction(group_identity) < candidate_probability
            )
            group_score = sum(scores[start:end]) / float(group_len)
            group_priorities.append(
                1.0 - group_score
                if admission.policy == "kv_score_low_only"
                else group_score
            )

        admitted_pages = sum(
            group_len
            for group_len, admitted in zip(group_lens, group_decisions)
            if admitted
        )
        if admission.min_replica_pages > 0 and admitted_pages == 0:
            for group_index in sorted(
                range(len(group_lens)),
                key=lambda index: group_priorities[index],
                reverse=True,
            ):
                group_decisions[group_index] = True
                admitted_pages += group_lens[group_index]
                if admitted_pages >= admission.min_replica_pages:
                    break

        max_replica = admission.max_replica_pages_per_batch
        if max_replica > 0 and admitted_pages > max_replica:
            keep: set[int] = set()
            kept_pages = 0
            for group_index in sorted(
                (
                    index
                    for index, admitted in enumerate(group_decisions)
                    if admitted
                ),
                key=lambda index: group_priorities[index],
                reverse=True,
            ):
                group_len = group_lens[group_index]
                if kept_pages + group_len > max_replica:
                    continue
                keep.add(group_index)
                kept_pages += group_len
            group_decisions = [
                group_index in keep for group_index in range(len(group_lens))
            ]

        mask: list[bool] = []
        for group_len, admitted in zip(group_lens, group_decisions):
            mask.extend([admitted] * group_len)
        if record_metrics:
            self._record_replica_task_admission(
                page_count,
                sum(1 for item in mask if item),
                op_name,
            )
        return mask

    @staticmethod
    def _slice_fragment_pages(
        page_indices: List[int],
        fragment_ptrs: List[int],
        fragment_lens: List[int],
        fragments_per_key: int,
    ) -> tuple[List[int], List[int]]:
        sliced_ptrs: list[int] = []
        sliced_lens: list[int] = []
        for page_index in page_indices:
            start_idx = page_index * fragments_per_key
            end_idx = start_idx + fragments_per_key
            sliced_ptrs.extend(fragment_ptrs[start_idx:end_idx])
            sliced_lens.extend(fragment_lens[start_idx:end_idx])
        return sliced_ptrs, sliced_lens

    def _put_result_to_bool(self, result: Any, storage_key: str, op_name: str) -> bool:
        if hasattr(result, "is_ok"):
            if result.is_ok():
                _ = result.unwrap()
                return True
            logger.warning(
                "Fluxon %s failed for %s: %s",
                op_name,
                storage_key,
                result.unwrap_error(),
            )
            return False
        if isinstance(result, bool):
            return result
        if isinstance(result, int):
            return result == 0
        return bool(result)

    def _fragment_ptr_meta_from_cpu_tensors(
        self, cpu_tensors: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[int], list[int], int]:
        if not cpu_tensors:
            raise RuntimeError("Fluxon fragment payload requires at least one fragment")
        payload_tensors: list[torch.Tensor] = []
        fragment_ptrs: list[int] = []
        fragment_lens: list[int] = []
        total_bytes = 0
        for cpu_tensor in cpu_tensors:
            if cpu_tensor.device.type != "cpu":
                cpu_tensor = cpu_tensor.to(device="cpu")
            if not cpu_tensor.is_contiguous():
                cpu_tensor = cpu_tensor.contiguous()
            payload_tensor = cpu_tensor.view(torch.uint8).reshape(-1)
            payload_tensors.append(payload_tensor)
            fragment_ptrs.append(int(payload_tensor.data_ptr()))
            fragment_len = int(payload_tensor.numel())
            fragment_lens.append(fragment_len)
            total_bytes += fragment_len
        return payload_tensors, fragment_ptrs, fragment_lens, total_bytes

    @staticmethod
    def _fragments_per_page(
        page_count: int,
        flat_values: List[int],
        value_name: str,
        op_name: str,
    ) -> int:
        if page_count == 0:
            if flat_values:
                raise RuntimeError(
                    f"Fluxon {op_name} got unexpected {value_name} for empty page batch"
                )
            return 0

        if len(flat_values) % page_count != 0:
            raise RuntimeError(
                f"Fluxon {op_name} cannot group {value_name}: total={len(flat_values)} "
                f"page_count={page_count}"
            )

        fragments_per_page = len(flat_values) // page_count
        if fragments_per_page == 0:
            raise RuntimeError(
                f"Fluxon {op_name} got zero fragments per page for {value_name}"
            )
        return fragments_per_page

    def _page_fragment_meta_from_host_indices(
        self,
        host_pool: Any,
        host_indices: torch.Tensor,
        page_count: int,
        op_name: str,
    ) -> tuple[List[int], List[int], int]:
        ptr_list, size_list = host_pool.get_page_buffer_meta(host_indices)
        ptr_values = [int(ptr) for ptr in ptr_list]
        size_values = [int(size) for size in size_list]
        if len(ptr_values) != len(size_values):
            raise RuntimeError(
                f"Fluxon {op_name} host pool returned ptr/size mismatch: "
                f"ptrs={len(ptr_values)} sizes={len(size_values)}"
            )

        fragments_per_page = self._fragments_per_page(
            page_count, ptr_values, "fragment_ptrs", op_name
        )
        size_fragments_per_page = self._fragments_per_page(
            page_count, size_values, "fragment_sizes", op_name
        )
        if fragments_per_page != size_fragments_per_page:
            raise RuntimeError(
                f"Fluxon {op_name} page fragment mismatch: ptrs_per_page="
                f"{fragments_per_page} sizes_per_page={size_fragments_per_page}"
            )
        return ptr_values, size_values, fragments_per_page

    def _batch_put_fragment_ptrs_flat(
        self,
        storage_keys: List[str],
        fragment_ptrs: List[int],
        fragment_lens: List[int],
        fragments_per_key: int,
        op_name: str,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> tuple[List[bool], int]:
        if not storage_keys:
            if fragment_ptrs or fragment_lens:
                raise ValueError(
                    f"{op_name} requires empty fragment_ptrs and fragment_lens when "
                    "storage_keys is empty"
                )
            return [], 0
        if fragments_per_key <= 0:
            raise ValueError(
                f"{op_name} requires fragments_per_key > 0 when storage_keys is non-empty"
            )
        expected_flat_len = len(storage_keys) * fragments_per_key
        if len(fragment_ptrs) != expected_flat_len or len(fragment_lens) != expected_flat_len:
            raise ValueError(
                f"{op_name} requires fragment_ptrs and fragment_lens to match "
                "len(storage_keys) * fragments_per_key"
            )

        batch_put = getattr(self.store, "batch_put_fragments_from_ptrs_blocking", None)
        if batch_put is None:
            raise RuntimeError(
                "Fluxon store does not expose batch_put_fragments_from_ptrs_blocking"
            )

        start = time.perf_counter()
        admission_mask = self._replica_task_admission_mask(
            storage_keys, extra_info, op_name
        )
        if len(admission_mask) != len(storage_keys):
            raise RuntimeError(
                f"Fluxon {op_name} admission mask length mismatch: "
                f"keys={len(storage_keys)} mask={len(admission_mask)}"
            )

        result_codes: list[int | None] = [None] * len(storage_keys)
        for make_replica_task in (True, False):
            page_indices = [
                index
                for index, admitted in enumerate(admission_mask)
                if admitted == make_replica_task
            ]
            if not page_indices:
                continue
            group_keys = [storage_keys[index] for index in page_indices]
            group_ptrs, group_lens = self._slice_fragment_pages(
                page_indices,
                fragment_ptrs,
                fragment_lens,
                fragments_per_key,
            )
            group_codes = batch_put(
                group_keys,
                group_ptrs,
                group_lens,
                fragments_per_key,
                opts=self._put_opts_for_replica_task(make_replica_task),
            )
            if not isinstance(group_codes, list) or len(group_codes) != len(group_keys):
                raise RuntimeError(
                    f"Fluxon {op_name} returned unexpected result payload for "
                    f"{len(group_keys)} keys"
                )
            for index, code in zip(page_indices, group_codes):
                result_codes[index] = code

        if any(code is None for code in result_codes):
            raise RuntimeError(f"Fluxon {op_name} did not populate all result slots")

        results: list[bool] = []
        success_bytes = 0
        for idx, (storage_key, code) in enumerate(zip(storage_keys, result_codes)):
            if not isinstance(code, int):
                raise RuntimeError(
                    f"Fluxon {op_name} returned non-int result for {storage_key!r}: "
                    f"{type(code)}"
                )
            ok = code == 0
            if ok:
                start_idx = idx * fragments_per_key
                success_bytes += sum(
                    fragment_lens[start_idx : start_idx + fragments_per_key]
                )
            else:
                logger.warning(
                    "Fluxon %s fragment put failed for %s: code=%s",
                    op_name,
                    storage_key,
                    code,
                )
            results.append(ok)

        logger.info(
            "Fluxon %s fragment put: pages=%d/%d bytes=%d duration_ms=%.3f",
            op_name,
            sum(1 for ok in results if ok),
            len(storage_keys),
            success_bytes,
            (time.perf_counter() - start) * 1000.0,
        )
        return results, success_bytes

    def _submit_batch_put_fragment_ptrs_flat(
        self,
        storage_keys: List[str],
        fragment_ptrs: List[int],
        fragment_lens: List[int],
        fragments_per_key: int,
        op_name: str,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Any:
        if not storage_keys:
            raise ValueError(f"{op_name} requires at least one storage key")
        if fragments_per_key <= 0:
            raise ValueError(
                f"{op_name} requires fragments_per_key > 0 when storage_keys is non-empty"
            )
        expected_flat_len = len(storage_keys) * fragments_per_key
        if len(fragment_ptrs) != expected_flat_len or len(fragment_lens) != expected_flat_len:
            raise ValueError(
                f"{op_name} requires fragment_ptrs and fragment_lens to match "
                "len(storage_keys) * fragments_per_key"
            )

        batch_put = getattr(self.store, "batch_put_fragments_from_ptrs", None)
        if batch_put is None:
            raise RuntimeError(
                "Fluxon store does not expose batch_put_fragments_from_ptrs"
            )

        admission_mask = self._replica_task_admission_mask(
            storage_keys,
            extra_info,
            op_name,
            record_metrics=False,
        )
        if len(admission_mask) != len(storage_keys):
            raise RuntimeError(
                f"Fluxon {op_name} admission mask length mismatch: "
                f"keys={len(storage_keys)} mask={len(admission_mask)}"
            )
        self._record_replica_task_admission(
            len(storage_keys),
            sum(1 for item in admission_mask if item),
            op_name,
        )
        group_futures: list[tuple[list[int], Any]] = []
        for make_replica_task in (True, False):
            page_indices = [
                index
                for index, admitted in enumerate(admission_mask)
                if admitted == make_replica_task
            ]
            if not page_indices:
                continue
            group_keys = [storage_keys[index] for index in page_indices]
            group_ptrs, group_lens = self._slice_fragment_pages(
                page_indices,
                fragment_ptrs,
                fragment_lens,
                fragments_per_key,
            )
            submit_result = batch_put(
                group_keys,
                group_ptrs,
                group_lens,
                fragments_per_key,
                opts=self._put_opts_for_replica_task(make_replica_task),
            )
            if hasattr(submit_result, "is_ok"):
                if not submit_result.is_ok():
                    err = submit_result.unwrap_error()
                    _wait_submitted_replica_groups(op_name, group_futures)
                    if isinstance(err, Exception):
                        raise err
                    raise RuntimeError(f"Fluxon {op_name} submit failed: {err}")
                submit_result = submit_result.unwrap()
            if submit_result is None:
                _wait_submitted_replica_groups(op_name, group_futures)
                raise RuntimeError(f"Fluxon {op_name} returned an empty future")
            group_futures.append((page_indices, submit_result))

        return _FluxonMergedBatchRetCodeFuture(
            op_name,
            len(storage_keys),
            group_futures,
        )

    def _batch_get_fragment_ptrs_flat(
        self,
        storage_keys: List[str],
        fragment_ptrs: List[int],
        fragment_capacities: List[int],
        fragments_per_key: int,
        op_name: str,
    ) -> tuple[List[bool], int]:
        if not storage_keys:
            if fragment_ptrs or fragment_capacities:
                raise ValueError(
                    f"{op_name} requires empty fragment_ptrs and fragment_capacities "
                    "when storage_keys is empty"
                )
            return [], 0
        if fragments_per_key <= 0:
            raise ValueError(
                f"{op_name} requires fragments_per_key > 0 when storage_keys is non-empty"
            )
        expected_flat_len = len(storage_keys) * fragments_per_key
        if (
            len(fragment_ptrs) != expected_flat_len
            or len(fragment_capacities) != expected_flat_len
        ):
            raise ValueError(
                f"{op_name} requires fragment_ptrs and fragment_capacities to match "
                "len(storage_keys) * fragments_per_key"
            )

        start = time.perf_counter()
        holder_or_flats = self._batch_get_values(storage_keys)
        if len(holder_or_flats) != len(storage_keys):
            raise RuntimeError(
                f"Fluxon {op_name} returned unexpected result payload for "
                f"{len(storage_keys)} keys"
            )

        results: list[bool] = []
        success_bytes = 0
        for key_idx, (storage_key, holder_or_flat) in enumerate(
            zip(storage_keys, holder_or_flats)
        ):
            if holder_or_flat is None:
                logger.warning(
                    "Fluxon %s fragment get missed page for %s",
                    op_name,
                    storage_key,
                )
                results.append(False)
                continue

            flat_offset = key_idx * fragments_per_key
            try:
                fragment_view = self._fragment_ptr_view_from_store_value(
                    storage_key, holder_or_flat
                )
                if len(fragment_view.fragment_ptrs) != fragments_per_key:
                    raise RuntimeError(
                        f"fragment count mismatch: expected={fragments_per_key} "
                        f"got={len(fragment_view.fragment_ptrs)}"
                    )

                copied_bytes = 0
                for fragment_idx, (src_ptr, src_len) in enumerate(
                    zip(fragment_view.fragment_ptrs, fragment_view.fragment_lens)
                ):
                    dst_ptr = int(fragment_ptrs[flat_offset + fragment_idx])
                    dst_capacity = int(
                        fragment_capacities[flat_offset + fragment_idx]
                    )
                    if src_len > dst_capacity:
                        raise RuntimeError(
                            f"fragment exceeds destination capacity: len={src_len} "
                            f"capacity={dst_capacity} fragment_idx={fragment_idx}"
                        )
                    if src_len > 0:
                        if src_ptr == 0:
                            raise RuntimeError(
                                f"source fragment pointer is zero for fragment_idx={fragment_idx}"
                            )
                        if dst_ptr == 0:
                            raise RuntimeError(
                                f"destination fragment pointer is zero for fragment_idx={fragment_idx}"
                            )
                        ctypes.memmove(dst_ptr, src_ptr, src_len)
                    copied_bytes += src_len

                success_bytes += copied_bytes
                results.append(True)
            except Exception as exc:
                logger.warning(
                    "Fluxon %s fragment get failed for %s: %s",
                    op_name,
                    storage_key,
                    exc,
                )
                results.append(False)

        logger.info(
            "Fluxon %s fragment get: pages=%d/%d bytes=%d duration_ms=%.3f",
            op_name,
            sum(1 for ok in results if ok),
            len(storage_keys),
            success_bytes,
            (time.perf_counter() - start) * 1000.0,
        )
        return results, success_bytes

    def _submit_warm_get(self, storage_key: str) -> bool:
        with self._warm_lock:
            if not self._enable_warm_get:
                return False
            if storage_key in self._warm_futures:
                return False
            if storage_key in self._warm_inflight:
                return False
            if len(self._warm_futures) + len(self._warm_inflight) >= self._warm_limit:
                return False
            self._warm_inflight.add(storage_key)

        try:
            get_result = self.store.get(storage_key)
            if not get_result.is_ok():
                _ = get_result.unwrap_error()
                return False

            future = get_result.unwrap()
            with self._warm_lock:
                if self._enable_warm_get:
                    if len(self._warm_futures) < self._warm_limit:
                        self._warm_futures.setdefault(storage_key, future)
                        return True
                return False
        finally:
            with self._warm_lock:
                self._warm_inflight.discard(storage_key)

    def _submit_warm_get_batch(self, storage_keys: list[str]) -> int:
        if not storage_keys or not hasattr(self.store, "batch_get_blocking"):
            return 0

        selected: list[str] = []
        with self._warm_lock:
            if not self._enable_warm_get:
                return 0
            remaining = self._warm_limit - (
                len(self._warm_futures) + len(self._warm_inflight)
            )
            if remaining <= 0:
                return 0
            for storage_key in storage_keys:
                if len(selected) >= remaining:
                    break
                if storage_key in self._warm_futures:
                    continue
                if storage_key in self._warm_inflight:
                    continue
                self._warm_inflight.add(storage_key)
                selected.append(storage_key)

        if not selected:
            return 0

        def _batch_get() -> Any:
            return self.store.batch_get_blocking(
                selected,
                concurrency=self._batch_concurrency,
            )

        try:
            batch_future = self._warm_batch_executor.submit(_batch_get)
        except Exception:
            with self._warm_lock:
                for storage_key in selected:
                    self._warm_inflight.discard(storage_key)
            raise

        accepted = 0
        with self._warm_lock:
            try:
                if not self._enable_warm_get:
                    return 0
                for batch_index, storage_key in enumerate(selected):
                    if len(self._warm_futures) >= self._warm_limit:
                        break
                    if storage_key in self._warm_futures:
                        continue
                    self._warm_futures[storage_key] = _FluxonBatchWarmFuture(
                        batch_future,
                        batch_index,
                        storage_key,
                    )
                    accepted += 1
                return accepted
            finally:
                for storage_key in selected:
                    self._warm_inflight.discard(storage_key)

    def _drain_ready_warm_futures(self, budget: Optional[int] = None) -> int:
        if budget is None:
            budget = self._warm_drain_budget
        if budget <= 0:
            return 0
        drained = 0
        ready_items: list[tuple[str, Any]] = []
        with self._warm_lock:
            for storage_key, future in list(self._warm_futures.items()):
                if drained >= budget:
                    break
                is_waiting = getattr(future, "is_waiting", None)
                if is_waiting is not None and is_waiting():
                    continue
                popped = self._warm_futures.pop(storage_key, None)
                if popped is not None:
                    ready_items.append((storage_key, popped))
                    drained += 1
        for storage_key, future in ready_items:
            wait_result = future.wait()
            if wait_result.is_ok():
                holder = wait_result.unwrap()
                del holder
            else:
                logger.debug(
                    "Fluxon warm get drain failed for %s: %s",
                    storage_key,
                    wait_result.unwrap_error(),
                )
        return drained

    def set_warm_get_enabled(
        self, enabled: bool, reason: Optional[str] = None
    ) -> None:
        enabled = bool(enabled)
        if self._enable_warm_get == enabled:
            return
        self._enable_warm_get = enabled
        if not enabled:
            # Keep unfinished owner-local transfers discoverable until their
            # holder is consumed; dropping them can race a later direct plan.
            self._drain_ready_warm_futures(budget=self._warm_limit)
        logger.info(
            "Fluxon warm_get %s%s",
            "enabled" if enabled else "disabled",
            f": {reason}" if reason else "",
        )

    def warm_get_enabled(self) -> bool:
        return self._enable_warm_get

    def _warm_keys(self, storage_keys: List[str]) -> int:
        if not self._enable_warm_get:
            return 0
        self._drain_ready_warm_futures()
        seen: set[str] = set()
        unique_keys: list[str] = []
        for storage_key in storage_keys:
            if len(unique_keys) >= self._warm_submit_limit:
                break
            if storage_key in seen:
                continue
            seen.add(storage_key)
            unique_keys.append(storage_key)

        submitted = self._submit_warm_get_batch(unique_keys)
        if submitted == 0 and not hasattr(self.store, "batch_get_blocking"):
            for storage_key in unique_keys:
                if submitted >= self._warm_submit_limit:
                    break
                if self._submit_warm_get(storage_key):
                    submitted += 1
        if submitted:
            with self._warm_lock:
                pending = len(self._warm_futures)
            logger.info(
                "Fluxon warm_get submitted: keys=%d pending=%d submit_limit=%d pending_limit=%d mode=%s",
                submitted,
                pending,
                self._warm_submit_limit,
                self._warm_limit,
                "batch" if hasattr(self.store, "batch_get_blocking") else "single",
            )
        return submitted

    def _take_warm_future(self, storage_key: str) -> Any | None:
        with self._warm_lock:
            return self._warm_futures.pop(storage_key, None)

    def _flat_dict_from_store_value(self, storage_key: str, holder_or_flat: Any) -> tuple[dict[str, Any], Any]:
        if hasattr(holder_or_flat, "access"):
            access_result = holder_or_flat.access()
            if hasattr(access_result, "is_ok"):
                if not access_result.is_ok():
                    raise RuntimeError(
                        f"Fluxon payload access failed for {storage_key!r}: "
                        f"{access_result.unwrap_error()}"
                    )
                flat = access_result.unwrap()
            else:
                flat = access_result
        elif isinstance(holder_or_flat, dict):
            flat = holder_or_flat
        else:
            raise RuntimeError(
                f"Fluxon get({storage_key!r}) returned unsupported type "
                f"{type(holder_or_flat)}"
            )

        if not isinstance(flat, dict):
            raise RuntimeError(
                f"Fluxon get({storage_key!r}) did not decode to a flat dict"
            )
        return flat, holder_or_flat

    def _cpu_payload_from_store_value(
        self, storage_key: str, holder_or_flat: Any
    ) -> tuple[torch.Tensor, Any]:
        flat, keepalive = self._flat_dict_from_store_value(storage_key, holder_or_flat)

        payload = flat.get("payload")
        if payload is None or not hasattr(payload, "__dlpack__"):
            raise RuntimeError(
                f"Fluxon payload for {storage_key!r} is not DLPack-capable"
            )

        payload_tensor = torch.from_dlpack(payload)
        if payload_tensor.device.type != "cpu":
            raise RuntimeError(
                f"Fluxon payload for {storage_key!r} must live on CPU, got "
                f"{payload_tensor.device}"
            )
        if not payload_tensor.is_contiguous():
            raise RuntimeError(
                f"Fluxon payload for {storage_key!r} must be CPU-contiguous"
            )
        return payload_tensor.view(torch.uint8).reshape(-1), keepalive

    def _fragment_ptr_view_from_store_value(
        self, storage_key: str, holder_or_flat: Any
    ) -> _FluxonFragmentPointerView:
        fragment_meta_fn = getattr(holder_or_flat, "fragment_ptrs_and_lens", None)
        if fragment_meta_fn is None:
            raise RuntimeError(
                f"Fluxon fragment pointer access for {storage_key!r} requires MemHolder fragment_ptrs_and_lens()"
            )
        fragment_meta_result = fragment_meta_fn()
        if hasattr(fragment_meta_result, "is_ok"):
            if not fragment_meta_result.is_ok():
                raise RuntimeError(
                    f"Fluxon fragment pointer access failed for {storage_key!r}: "
                    f"{fragment_meta_result.unwrap_error()}"
                )
            fragment_meta = fragment_meta_result.unwrap()
        else:
            fragment_meta = fragment_meta_result

        if not isinstance(fragment_meta, tuple) or len(fragment_meta) != 2:
            raise RuntimeError(
                f"Fluxon fragment pointer access for {storage_key!r} returned invalid payload "
                f"{type(fragment_meta)}"
            )
        fragment_ptrs_raw, fragment_lens_raw = fragment_meta
        fragment_ptrs = [int(ptr) for ptr in fragment_ptrs_raw]
        fragment_lens = [int(length) for length in fragment_lens_raw]
        if len(fragment_ptrs) == 0:
            raise RuntimeError(
                f"Fluxon fragment pointer access for {storage_key!r} returned zero fragments"
            )
        if len(fragment_ptrs) != len(fragment_lens):
            raise RuntimeError(
                f"Fluxon fragment pointer access for {storage_key!r} length mismatch: "
                f"ptrs={len(fragment_ptrs)} lens={len(fragment_lens)}"
            )
        return _FluxonFragmentPointerView(fragment_ptrs, fragment_lens, holder_or_flat)

    def _blocking_get_value(self, storage_key: str) -> Any:
        get_result = self.store.get_blocking(storage_key)
        if not get_result.is_ok():
            raise RuntimeError(
                f"Fluxon get_blocking({storage_key!r}) failed: "
                f"{get_result.unwrap_error()}"
            )
        return get_result.unwrap()

    def _get_value(self, storage_key: str) -> Any:
        future = self._take_warm_future(storage_key)
        if future is not None:
            waited = self._wait_warm_value(storage_key, future)
            if waited is not None:
                return waited
        return self._blocking_get_value(storage_key)

    def _wait_warm_value(self, storage_key: str, future: Any) -> Any | None:
        wait_result = future.wait()
        if wait_result.is_ok():
            return wait_result.unwrap()
        logger.debug(
            "Fluxon warm get failed for %s, fallback to get_blocking: %s",
            storage_key,
            wait_result.unwrap_error(),
        )
        return None

    def _batch_get_values(self, storage_keys: List[str]) -> List[Any | None]:
        if not storage_keys:
            return []

        total_start = time.perf_counter()
        if not hasattr(self.store, "batch_get_blocking"):
            results = [self._get_value(storage_key) for storage_key in storage_keys]
            logger.info(
                "Fluxon _batch_get_values sequential: keys=%d duration_ms=%.3f",
                len(storage_keys),
                (time.perf_counter() - total_start) * 1000.0,
            )
            return results

        values: list[Any | None] = [None] * len(storage_keys)
        warm_hits = 0
        cold_indices: list[int] = []
        cold_keys: list[str] = []
        for idx, storage_key in enumerate(storage_keys):
            future = self._take_warm_future(storage_key)
            if future is None:
                cold_indices.append(idx)
                cold_keys.append(storage_key)
                continue
            waited = self._wait_warm_value(storage_key, future)
            if waited is None:
                cold_indices.append(idx)
                cold_keys.append(storage_key)
                continue
            values[idx] = waited
            warm_hits += 1

        if not cold_keys:
            logger.info(
                "Fluxon _batch_get_values warm-only: keys=%d warm_hits=%d duration_ms=%.3f",
                len(storage_keys),
                warm_hits,
                (time.perf_counter() - total_start) * 1000.0,
            )
            return values

        call_start = time.perf_counter()
        batch_results = self.store.batch_get_blocking(
            cold_keys,
            concurrency=self._batch_concurrency,
        )
        call_duration_ms = (time.perf_counter() - call_start) * 1000.0
        if hasattr(batch_results, "is_ok"):
            if not batch_results.is_ok():
                logger.warning(
                    "Fluxon batch_get_blocking failed for %d cold keys: %s; fallback to sequential get",
                    len(cold_keys),
                    batch_results.unwrap_error(),
                )
                for idx, storage_key in zip(cold_indices, cold_keys):
                    values[idx] = self._blocking_get_value(storage_key)
                return values
            batch_results = batch_results.unwrap()

        if not isinstance(batch_results, list) or len(batch_results) != len(cold_keys):
            logger.warning(
                "Fluxon batch_get_blocking returned unexpected payload for %d cold keys; fallback to sequential get",
                len(cold_keys),
            )
            for idx, storage_key in zip(cold_indices, cold_keys):
                values[idx] = self._blocking_get_value(storage_key)
            return values

        parse_start = time.perf_counter()
        for idx, storage_key, result in zip(cold_indices, cold_keys, batch_results):
            if hasattr(result, "is_ok"):
                if result.is_ok():
                    values[idx] = result.unwrap()
                else:
                    logger.warning(
                        "Fluxon batch_get_blocking failed for %s: %s",
                        storage_key,
                        result.unwrap_error(),
                    )
                    values[idx] = None
            else:
                values[idx] = result
        logger.info(
            "Fluxon _batch_get_values: keys=%d warm_hits=%d cold_keys=%d hits=%d misses=%d concurrency=%d "
            "batch_get_blocking_ms=%.3f result_parse_ms=%.3f duration_ms=%.3f",
            len(storage_keys),
            warm_hits,
            len(cold_keys),
            sum(1 for value in values if value is not None),
            sum(1 for value in values if value is None),
            self._batch_concurrency,
            call_duration_ms,
            (time.perf_counter() - parse_start) * 1000.0,
            (time.perf_counter() - total_start) * 1000.0,
        )
        return values

    @staticmethod
    def _restore_tensor_from_payload(
        payload_bytes: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        target_bytes = target_tensor.view(torch.uint8).reshape(-1)
        if target_bytes.numel() != payload_bytes.numel():
            raise RuntimeError(
                "Fluxon payload size mismatch: "
                f"expected={target_bytes.numel()} got={payload_bytes.numel()}"
            )
        target_bytes.copy_(payload_bytes)
        return target_tensor

    def local_fast_put_start(
        self,
        keys: List[str],
        value_len: int,
        component_name: Optional[Any] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        if not keys:
            raise ValueError("Fluxon local_fast_put_start requires at least one key")
        if value_len <= 0:
            raise ValueError(
                f"Fluxon local_fast_put_start requires value_len > 0, got {value_len}"
            )
        self._assert_fluxon_cuda_segments_registered("local_fast_put_start")
        storage_keys = [self._store_key(key, component_name) for key in keys]
        atomic_group_lens = self._normalize_atomic_group_lens(
            storage_keys, extra_info
        )
        admission_mask = self._replica_task_admission_mask(
            storage_keys,
            extra_info,
            "local_fast_put_start",
        )
        radix_parent_keys = self._radix_parent_keys(
            storage_keys, extra_info, component_name
        )
        content_depths = self._absolute_content_depths(storage_keys, extra_info)
        start = time.perf_counter()
        plan_ptr = self._call_local_fast_put_start(
            storage_keys,
            value_len,
            opts=self._put_opts_for_replica_task_mask(
                admission_mask,
                radix_parent_keys,
                content_depths,
                atomic_group_lens,
            ),
            make_replica_task=any(admission_mask),
            admission_mask=admission_mask,
            radix_parent_keys=radix_parent_keys,
            content_depths=content_depths,
            atomic_group_lens=atomic_group_lens,
        )
        logger.info(
            "Fluxon local_fast_put_start success: pages=%d value_len=%d "
            "component=%s replica_admitted=%d decision_scope=atomic_group "
            "content_depth_min=%s content_depth_max=%s "
            "atomic_group_lens=%s duration_ms=%.3f",
            len(storage_keys),
            value_len,
            component_name,
            sum(1 for item in admission_mask if item),
            None if content_depths is None else content_depths[0],
            None if content_depths is None else content_depths[-1],
            atomic_group_lens,
            (time.perf_counter() - start) * 1000.0,
        )
        return int(plan_ptr)

    def local_fast_put_start_replica_admitted_count(
        self,
        keys: List[str],
        component_name: Optional[Any] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        if not keys:
            raise ValueError(
                "Fluxon local_fast_put_start_replica_admitted_count requires at least one key"
            )
        storage_keys = [self._store_key(key, component_name) for key in keys]
        admission_mask = self._replica_task_admission_mask(
            storage_keys,
            extra_info,
            "local_fast_put_start_replica_admitted_count",
            record_metrics=False,
        )
        return sum(1 for item in admission_mask if item)

    def local_fast_put_start_local_only(
        self,
        keys: List[str],
        value_len: int,
        component_name: Optional[Any] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        if not keys:
            raise ValueError(
                "Fluxon local_fast_put_start_local_only requires at least one key"
            )
        if value_len <= 0:
            raise ValueError(
                "Fluxon local_fast_put_start_local_only requires value_len > 0, "
                f"got {value_len}"
            )
        self._assert_fluxon_cuda_segments_registered("local_fast_put_start_local_only")
        storage_keys = [self._store_key(key, component_name) for key in keys]
        atomic_group_lens = self._normalize_atomic_group_lens(
            storage_keys, extra_info
        )
        admission_mask = [False] * len(storage_keys)
        radix_parent_keys = self._radix_parent_keys(
            storage_keys, extra_info, component_name
        )
        content_depths = self._absolute_content_depths(storage_keys, extra_info)
        start = time.perf_counter()
        plan_ptr = self._call_local_fast_put_start(
            storage_keys,
            value_len,
            opts=self._put_opts_for_replica_task_mask(
                admission_mask,
                radix_parent_keys,
                content_depths,
                atomic_group_lens,
            ),
            make_replica_task=False,
            admission_mask=admission_mask,
            radix_parent_keys=radix_parent_keys,
            content_depths=content_depths,
            atomic_group_lens=atomic_group_lens,
        )
        logger.info(
            "Fluxon local_fast_put_start_local_only success: pages=%d value_len=%d "
            "component=%s replica_admitted=0 duration_ms=%.3f",
            len(storage_keys),
            value_len,
            component_name,
            (time.perf_counter() - start) * 1000.0,
        )
        return int(plan_ptr)

    def local_fast_put_commit(self, plan_ptr: int) -> Any:
        start = time.perf_counter()
        future = self.store.local_fast_put_commit(plan_ptr)
        logger.info(
            "Fluxon local_fast_put_commit submitted: plan_ptr=%#x duration_ms=%.3f",
            int(plan_ptr),
            (time.perf_counter() - start) * 1000.0,
        )
        return future

    def put_abort(self, plan_ptr: int) -> None:
        start = time.perf_counter()
        self.store.put_abort(plan_ptr)
        logger.info(
            "Fluxon put_abort complete: plan_ptr=%#x duration_ms=%.3f",
            int(plan_ptr),
            (time.perf_counter() - start) * 1000.0,
        )

    def get_views(
        self,
        keys: List[str],
        component_name: Optional[Any] = None,
    ) -> int:
        if not keys:
            raise ValueError("Fluxon get_views requires at least one key")
        self._assert_fluxon_cuda_segments_registered("get_views")
        storage_keys = [self._store_key(key, component_name) for key in keys]
        start = time.perf_counter()
        plan_ptr = self.store.get_views(
            storage_keys,
            concurrency=self._batch_concurrency,
        )
        logger.info(
            "Fluxon get_views success: pages=%d component=%s concurrency=%d duration_ms=%.3f",
            len(storage_keys),
            component_name,
            self._batch_concurrency,
            (time.perf_counter() - start) * 1000.0,
        )
        return int(plan_ptr)

    def get_start(
        self,
        keys: List[str],
        component_name: Optional[Any] = None,
        prefix_best_effort: bool = True,
        atomic_group_lens: Optional[List[int]] = None,
    ) -> Any:
        if not keys:
            raise ValueError("Fluxon get_start requires at least one key")
        self._assert_fluxon_cuda_segments_registered("get_start")
        storage_keys = [self._store_key(key, component_name) for key in keys]
        start = time.perf_counter()
        handle = self.store.get_start(
            storage_keys,
            prefix_best_effort=prefix_best_effort,
            atomic_group_lens=atomic_group_lens,
        )
        result = handle.result
        logger.info(
            "Fluxon get_start success: pages=%d raw_prefix_hit_len=%s "
            "transferable_len=%s atomic_group_lens=%s all_hit=%s component=%s "
            "concurrency=%d duration_ms=%.3f",
            len(storage_keys),
            getattr(result, "raw_prefix_hit_len", None),
            getattr(result, "transferable_len", None),
            atomic_group_lens,
            getattr(result, "all_hit", None),
            component_name,
            self._batch_concurrency,
            (time.perf_counter() - start) * 1000.0,
        )
        return handle

    def configure_gpu_direct_staging(
        self,
        value_len: int,
        slot_count: int,
        device_id: int,
    ) -> None:
        if value_len <= 0 or slot_count <= 0:
            raise ValueError(
                "Fluxon GPU staging requires positive value_len/slot_count: "
                f"value_len={value_len} slot_count={slot_count}"
            )
        existing = self._gpu_direct_staging_pool
        if existing is not None:
            if (
                existing.slot_size != int(value_len)
                or existing.slot_count != int(slot_count)
                or existing.device_id != int(device_id)
            ):
                raise RuntimeError(
                    "Fluxon GPU staging cannot be reconfigured: "
                    f"existing=({existing.slot_size},{existing.slot_count},{existing.device_id}) "
                    f"requested=({value_len},{slot_count},{device_id})"
                )
            return
        for method_name in (
            "register_gpu_buffer",
            "get_plan",
            "execute_get_plan_cpu",
            "execute_get_plan_gpu",
            "cancel_get_plan",
            "get_start_gpu",
            "get_transfer_gpu",
            "cancel_get_transfer_gpu",
        ):
            if not hasattr(self.store, method_name):
                raise RuntimeError(
                    f"Fluxon store lacks GPU-direct method {method_name}"
                )
        self._gpu_direct_staging_pool = _FluxonGpuStagingPool(
            self.store,
            int(value_len),
            int(slot_count),
            int(device_id),
            self._fixed_slab_allocator_type,
        )

    def try_reserve_gpu_direct_staging(
        self,
        page_count: int,
        admission_block_reason: Optional[str] = None,
    ) -> tuple[Optional[_FluxonGpuStagingLease], dict[str, Any]]:
        pool = self._gpu_direct_staging_pool
        if pool is None:
            reason = (
                str(admission_block_reason)
                if admission_block_reason is not None
                else "pool_unconfigured"
            )
            return None, {
                "reason": reason,
                "requested_pages": int(page_count),
                "capacity_slots": 0,
                "free_slots_before": 0,
                "live_slots_before": 0,
                "active_leases_before": 0,
                "free_slots_after": 0,
                "live_slots_after": 0,
                "active_leases_after": 0,
                "high_watermark_slots": 0,
            }
        return pool.try_reserve(
            int(page_count),
            admission_block_reason=admission_block_reason,
        )

    def get_plan(
        self,
        keys: List[str],
        component_name: Optional[Any] = None,
        prefix_best_effort: bool = True,
        atomic_group_lens: Optional[List[int]] = None,
    ) -> Any:
        if not keys:
            raise ValueError("Fluxon get_plan requires at least one key")
        storage_keys = [self._store_key(key, component_name) for key in keys]
        start = time.perf_counter()
        handle = self.store.get_plan(
            storage_keys,
            prefix_best_effort=prefix_best_effort,
            atomic_group_lens=atomic_group_lens,
        )
        logger.info(
            "Fluxon get_plan success: pages=%d cpu_transferable=%s "
            "gpu_transferable=%s atomic_group_lens=%s component=%s duration_ms=%.3f",
            len(storage_keys),
            getattr(handle.result, "transferable_len", None),
            getattr(handle.gpu_result, "transferable_len", None),
            atomic_group_lens,
            component_name,
            (time.perf_counter() - start) * 1000.0,
        )
        return handle

    def cancel_get_plan(self, handle: Any) -> None:
        if handle.closed:
            return
        self.store.cancel_get_plan(handle)

    def execute_get_plan_cpu(
        self,
        handle: Any,
        *,
        consume_prefix_len: int,
    ) -> Any:
        start = time.perf_counter()
        executed = self.store.execute_get_plan_cpu(
            handle,
            consume_prefix_len=int(consume_prefix_len),
            concurrency=self._batch_concurrency,
        )
        logger.info(
            "Fluxon execute_get_plan_cpu success: consume_prefix_len=%d duration_ms=%.3f",
            int(consume_prefix_len),
            (time.perf_counter() - start) * 1000.0,
        )
        return executed

    def execute_get_plan_gpu(
        self,
        handle: Any,
        staging_lease: _FluxonGpuStagingLease,
        *,
        consume_prefix_len: int,
    ) -> Any:
        remote_count = sum(
            int(index) < int(consume_prefix_len)
            for index in handle.gpu_remote_indices
        )
        if (
            staging_lease.released
            or remote_count <= 0
            or staging_lease.page_count != remote_count
        ):
            raise RuntimeError(
                "Fluxon execute_get_plan_gpu staging/remote mismatch: "
                f"released={staging_lease.released} slots={staging_lease.page_count} "
                f"remote={remote_count} consume={consume_prefix_len}"
            )
        start = time.perf_counter()
        executed = self.store.execute_get_plan_gpu(
            handle,
            list(staging_lease.destinations),
            consume_prefix_len=int(consume_prefix_len),
            concurrency=self._batch_concurrency,
        )
        logger.info(
            "Fluxon execute_get_plan_gpu success: consume_prefix_len=%d duration_ms=%.3f",
            int(consume_prefix_len),
            (time.perf_counter() - start) * 1000.0,
        )
        return executed

    def get_start_gpu(
        self,
        keys: List[str],
        staging_lease: _FluxonGpuStagingLease,
        component_name: Optional[Any] = None,
        prefix_best_effort: bool = True,
        atomic_group_lens: Optional[List[int]] = None,
    ) -> Any:
        if not keys:
            raise ValueError("Fluxon get_start_gpu requires at least one key")
        if staging_lease.released or staging_lease.page_count != len(keys):
            raise RuntimeError(
                "Fluxon get_start_gpu staging/key mismatch: "
                f"released={staging_lease.released} slots={staging_lease.page_count} "
                f"keys={len(keys)}"
            )
        storage_keys = [self._store_key(key, component_name) for key in keys]
        start = time.perf_counter()
        handle = self.store.get_start_gpu(
            storage_keys,
            list(staging_lease.destinations),
            prefix_best_effort=prefix_best_effort,
            atomic_group_lens=atomic_group_lens,
        )
        result = handle.result
        logger.info(
            "Fluxon get_start_gpu success: pages=%d raw_prefix_hit_len=%s "
            "transferable_len=%s atomic_group_lens=%s all_hit=%s component=%s "
            "concurrency=%d duration_ms=%.3f",
            len(storage_keys),
            getattr(result, "raw_prefix_hit_len", None),
            getattr(result, "transferable_len", None),
            atomic_group_lens,
            getattr(result, "all_hit", None),
            component_name,
            self._batch_concurrency,
            (time.perf_counter() - start) * 1000.0,
        )
        return handle

    def cancel_get_transfer_gpu(self, handle: Any) -> None:
        if handle.closed:
            return
        start = time.perf_counter()
        self.store.cancel_get_transfer_gpu(handle)
        logger.info(
            "Fluxon cancel_get_transfer_gpu success: duration_ms=%.3f",
            (time.perf_counter() - start) * 1000.0,
        )

    def get_transfer_gpu(
        self,
        handle: Any,
        *,
        consume_prefix_len: Optional[int] = None,
    ) -> int:
        if handle.closed:
            raise RuntimeError("Fluxon get_transfer_gpu requires an open handle")
        start = time.perf_counter()
        plan_ptr = self.store.get_transfer_gpu(
            handle,
            consume_prefix_len=consume_prefix_len,
        )
        logger.info(
            "Fluxon get_transfer_gpu success: plan_ptr=%#x consume_prefix_len=%s "
            "backend_handle=%s transfer_wall_us=%s terminal_before_consume=%s "
            "terminal_to_consume_us=%s finish_wait_us=%s duration_ms=%.3f",
            int(plan_ptr),
            consume_prefix_len,
            getattr(handle, "backend_handle", None),
            getattr(handle, "transfer_wall_us", None),
            getattr(handle, "terminal_before_consume", None),
            getattr(handle, "terminal_to_consume_us", None),
            getattr(handle, "finish_wait_us", None),
            (time.perf_counter() - start) * 1000.0,
        )
        return int(plan_ptr)

    def cancel_get_transfer(self, handle: Any) -> None:
        if handle.closed:
            return
        start = time.perf_counter()
        self.store.cancel_get_transfer(handle)
        logger.info(
            "Fluxon cancel_get_transfer success: duration_ms=%.3f",
            (time.perf_counter() - start) * 1000.0,
        )

    def get_transfer(
        self,
        handle: Any,
        *,
        consume_prefix_len: Optional[int] = None,
    ) -> int:
        if handle.closed:
            raise RuntimeError("Fluxon get_transfer requires an open handle")
        start = time.perf_counter()
        plan_ptr = self.store.get_transfer(
            handle,
            concurrency=self._batch_concurrency,
            consume_prefix_len=consume_prefix_len,
        )
        logger.info(
            "Fluxon get_transfer success: plan_ptr=%#x duration_ms=%.3f",
            int(plan_ptr),
            (time.perf_counter() - start) * 1000.0,
        )
        return int(plan_ptr)

    def release_views(self, plan_ptr: int) -> None:
        start = time.perf_counter()
        self.store.release_views(plan_ptr)
        logger.info(
            "Fluxon release_views complete: plan_ptr=%#x duration_ms=%.3f",
            int(plan_ptr),
            (time.perf_counter() - start) * 1000.0,
        )

    def view_value_ptrs(
        self,
        plan_ptr: int,
        expected_count: int,
    ) -> tuple[int, ...]:
        if plan_ptr <= 0:
            raise ValueError(f"plan_ptr must be > 0, got {plan_ptr}")
        if expected_count < 0:
            raise ValueError(
                f"expected_count must be >= 0, got {expected_count}"
            )
        header = (ctypes.c_uint64 * 2).from_address(plan_ptr)
        if int(header[0]) != _FLUXON_PLAN_BLOB_MAGIC:
            raise RuntimeError(
                "Invalid Fluxon plan blob magic: "
                f"expected={_FLUXON_PLAN_BLOB_MAGIC:#x} got={int(header[0]):#x}"
            )
        value_count = int(header[1])
        if value_count < expected_count:
            raise RuntimeError(
                "Fluxon plan does not cover the requested prefix: "
                f"requested={expected_count} available={value_count}"
            )
        values = (ctypes.c_uint64 * expected_count).from_address(plan_ptr + 16)
        value_ptrs = tuple(int(value) for value in values)
        if any(value_ptr == 0 for value_ptr in value_ptrs):
            raise RuntimeError("Fluxon plan contains a null value pointer")
        return value_ptrs

    def _put_payload(self, storage_key: str, cpu_tensor: torch.Tensor) -> bool:
        total_start = time.perf_counter()
        prep_start = time.perf_counter()
        if cpu_tensor.device.type != "cpu":
            cpu_tensor = cpu_tensor.to(device="cpu")
        if not cpu_tensor.is_contiguous():
            cpu_tensor = cpu_tensor.contiguous()
        payload_tensor = cpu_tensor.view(torch.uint8).reshape(-1)
        prep_ms = (time.perf_counter() - prep_start) * 1000.0
        put_start = time.perf_counter()
        put_result = self.store.put_blocking(
            storage_key,
            {"payload": payload_tensor},
            opts=self._put_opts_for_replica_task(
                self._replica_task_admission_mask(
                    [storage_key],
                    None,
                    "put_payload",
                )[0]
            ),
        )
        put_ms = (time.perf_counter() - put_start) * 1000.0
        if put_result.is_ok():
            _ = put_result.unwrap()
            logger.info(
                "Fluxon put_payload success: key=%s bytes=%d "
                "payload_prep_ms=%.3f put_blocking_ms=%.3f duration_ms=%.3f",
                storage_key,
                int(payload_tensor.numel()),
                prep_ms,
                put_ms,
                (time.perf_counter() - total_start) * 1000.0,
            )
            return True
        logger.warning(
            "Fluxon put_blocking failed for %s: payload_prep_ms=%.3f "
            "put_blocking_ms=%.3f duration_ms=%.3f error=%s",
            storage_key,
            prep_ms,
            put_ms,
            (time.perf_counter() - total_start) * 1000.0,
            put_result.unwrap_error(),
        )
        return False

    def _batch_put_payloads(
        self,
        storage_keys: List[str],
        cpu_tensors: List[torch.Tensor],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if not storage_keys:
            return []
        if len(storage_keys) != len(cpu_tensors):
            raise ValueError("batch put requires storage_keys and cpu_tensors to have the same length")

        if hasattr(self.store, "batch_put_blocking"):
            total_start = time.perf_counter()
            prep_start = time.perf_counter()
            flat_values: list[dict[str, torch.Tensor]] = []
            total_bytes = 0
            for cpu_tensor in cpu_tensors:
                if cpu_tensor.device.type != "cpu":
                    cpu_tensor = cpu_tensor.to(device="cpu")
                if not cpu_tensor.is_contiguous():
                    cpu_tensor = cpu_tensor.contiguous()
                payload_tensor = cpu_tensor.view(torch.uint8).reshape(-1)
                total_bytes += int(payload_tensor.numel())
                flat_values.append({"payload": payload_tensor})
            prep_ms = (time.perf_counter() - prep_start) * 1000.0

            put_start = time.perf_counter()
            admission_mask = self._replica_task_admission_mask(
                storage_keys, extra_info, "batch_put_blocking"
            )
            batch_results: list[Any | None] = [None] * len(storage_keys)
            batch_failed_error: Any | None = None
            for make_replica_task in (True, False):
                item_indices = [
                    index
                    for index, admitted in enumerate(admission_mask)
                    if admitted == make_replica_task
                ]
                if not item_indices:
                    continue
                group_results = self.store.batch_put_blocking(
                    [storage_keys[index] for index in item_indices],
                    [flat_values[index] for index in item_indices],
                    opts=self._put_opts_for_replica_task(make_replica_task),
                    concurrency=self._batch_concurrency,
                )
                if hasattr(group_results, "is_ok"):
                    if not group_results.is_ok():
                        batch_failed_error = group_results.unwrap_error()
                        break
                    group_results = group_results.unwrap()
                if not isinstance(group_results, list) or len(group_results) != len(item_indices):
                    batch_failed_error = (
                        "batch_put_blocking returned unexpected payload for "
                        f"{len(item_indices)} keys"
                    )
                    break
                for index, result in zip(item_indices, group_results):
                    batch_results[index] = result
            put_ms = (time.perf_counter() - put_start) * 1000.0
            if batch_failed_error is not None:
                logger.warning(
                    "Fluxon batch_put_blocking failed for %d keys: "
                    "payload_prep_ms=%.3f batch_put_blocking_ms=%.3f "
                    "duration_ms=%.3f error=%s",
                    len(storage_keys),
                    prep_ms,
                    put_ms,
                    (time.perf_counter() - total_start) * 1000.0,
                    batch_failed_error,
                )
                return [
                    False
                    if result is None
                    else self._put_result_to_bool(
                        result,
                        storage_key,
                        "batch_put_blocking",
                    )
                    for storage_key, result in zip(storage_keys, batch_results)
                ]

            if isinstance(batch_results, list) and len(batch_results) == len(storage_keys):
                parse_start = time.perf_counter()
                results: list[bool] = []
                for storage_key, result in zip(storage_keys, batch_results):
                    if result is None:
                        raise RuntimeError(
                            f"Fluxon batch_put_blocking did not populate result for {storage_key}"
                        )
                    results.append(
                        self._put_result_to_bool(
                            result, storage_key, "batch_put_blocking"
                        )
                    )
                parse_ms = (time.perf_counter() - parse_start) * 1000.0
                logger.info(
                    "Fluxon _batch_put_payloads success: pages=%d bytes=%d "
                    "concurrency=%d payload_prep_ms=%.3f batch_put_blocking_ms=%.3f "
                    "result_parse_ms=%.3f duration_ms=%.3f",
                    sum(1 for ok in results if ok),
                    total_bytes,
                    self._batch_concurrency,
                    prep_ms,
                    put_ms,
                    parse_ms,
                    (time.perf_counter() - total_start) * 1000.0,
                )
                return results

            logger.warning(
                "Fluxon batch_put_blocking returned unexpected payload for %d keys: "
                "payload_prep_ms=%.3f batch_put_blocking_ms=%.3f duration_ms=%.3f; "
                "fallback to sequential put",
                len(storage_keys),
                prep_ms,
                put_ms,
                (time.perf_counter() - total_start) * 1000.0,
            )

        return [
            self._put_payload(storage_key, tensor)
            for storage_key, tensor in zip(storage_keys, cpu_tensors)
        ]

    def _read_page(self, pool_name: Any, key: str, host_pool, page_offset: int) -> bool:
        storage_key = self._store_key(key, pool_name)
        try:
            holder_or_flat = self._get_value(storage_key)
            payload_bytes, keepalive = self._cpu_payload_from_store_value(
                storage_key, holder_or_flat
            )
            restored_page = self._restore_tensor_from_payload(
                payload_bytes, host_pool.get_dummy_flat_data_page()
            )
            host_pool.set_from_flat_data_page(page_offset, restored_page)
            _ = keepalive
            logger.info(
                "Fluxon read_page success: key=%s pool=%s bytes=%d",
                storage_key,
                pool_name,
                int(payload_bytes.numel()),
            )
            return True
        except Exception as exc:
            logger.warning("Fluxon page read failed for %s: %s", storage_key, exc)
            return False

    def _write_page(self, pool_name: Any, key: str, host_pool, page_offset: int) -> bool:
        storage_key = self._store_key(key, pool_name)
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self._put_payload(storage_key, data_page)

    @staticmethod
    def _page_offsets(
        keys: List[str], host_pool: Any, host_indices: torch.Tensor, op_name: str
    ) -> Optional[List[int]]:
        page_size = int(host_pool.page_size)
        expected = len(keys) * page_size
        if host_indices.numel() != expected:
            logger.error(
                "Fluxon %s indices length mismatch: expected=%s got=%s",
                op_name,
                expected,
                host_indices.numel(),
            )
            return None

        page_offsets: list[int] = []
        for page_indices in host_indices.reshape(len(keys), page_size).tolist():
            page_offset = int(page_indices[0])
            if page_indices != list(range(page_offset, page_offset + page_size)):
                logger.error(
                    "Fluxon %s requires contiguous host pages: indices=%s",
                    op_name,
                    page_indices,
                )
                return None
            page_offsets.append(page_offset)
        return page_offsets

    def _batch_read_pages(
        self, storage_keys: List[str], host_pool: Any, page_offsets: List[int]
    ) -> tuple[List[bool], int]:
        holder_or_flats = self._batch_get_values(storage_keys)
        results: list[bool] = []
        total_bytes = 0
        for storage_key, page_offset, holder_or_flat in zip(
            storage_keys, page_offsets, holder_or_flats
        ):
            if holder_or_flat is None:
                results.append(False)
                continue
            try:
                payload_bytes, keepalive = self._cpu_payload_from_store_value(
                    storage_key, holder_or_flat
                )
                restored_page = self._restore_tensor_from_payload(
                    payload_bytes, host_pool.get_dummy_flat_data_page()
                )
                host_pool.set_from_flat_data_page(page_offset, restored_page)
                total_bytes += int(payload_bytes.numel())
                _ = keepalive
                results.append(True)
            except Exception as exc:
                logger.warning("Fluxon page read failed for %s: %s", storage_key, exc)
                results.append(False)
        return results, total_bytes

    def _batch_write_pages(
        self,
        storage_keys: List[str],
        host_pool: Any,
        page_offsets: List[int],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> tuple[List[bool], int]:
        pages = [
            host_pool.get_data_page(page_offset, flat=True)
            for page_offset in page_offsets
        ]
        results = self._batch_put_payloads(storage_keys, pages, extra_info)
        total_bytes = sum(
            int(page.numel() * page.element_size())
            for page, succeeded in zip(pages, results)
            if succeeded
        )
        return results, total_bytes

    def _batch_io_v2(
        self,
        transfers: List[PoolTransfer],
        op_name: str,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ):
        results: dict[str, List[bool]] = {}
        start = time.perf_counter()
        total_pages = 0
        total_bytes = 0

        for transfer in transfers:
            host_pool = getattr(self, "registered_pools", {}).get(transfer.name)
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) if host_pool else 1
            expected = len(keys) * page_size
            host_indices = transfer.host_indices

            if host_pool is None:
                logger.error("Fluxon v2 pool %s is not registered", transfer.name)
                results[transfer.name] = [False] * len(keys)
                continue

            if host_indices is None or host_indices.numel() != expected:
                logger.error(
                    "Fluxon %s indices length mismatch for %s: expected=%s got=%s",
                    op_name,
                    transfer.name,
                    expected,
                    host_indices.numel() if host_indices is not None else 0,
                )
                results[transfer.name] = [False] * len(keys)
                continue

            page_offsets = self._page_offsets(
                keys, host_pool, host_indices, f"{op_name}/{transfer.name}"
            )
            if page_offsets is None:
                results[transfer.name] = [False] * len(keys)
                continue
            storage_keys = [self._store_key(key, transfer.name) for key in keys]
            if op_name == "batch_get_v2":
                pool_results, pool_bytes = self._batch_read_pages(
                    storage_keys, host_pool, page_offsets
                )
                total_pages += sum(1 for ok in pool_results if ok)
                total_bytes += pool_bytes
                results[transfer.name] = pool_results
            else:
                pool_results, put_bytes = self._batch_write_pages(
                    storage_keys,
                    host_pool,
                    page_offsets,
                    extra_info,
                )
                total_pages += sum(1 for ok in pool_results if ok)
                total_bytes += put_bytes
                results[transfer.name] = pool_results

        if op_name == "batch_get_v2":
            self._record_prefetch_metrics(total_pages, total_bytes, start)
        else:
            self._record_backup_metrics(total_pages, total_bytes, start)
        if total_pages > 0:
            logger.info(
                "Fluxon %s success: pages=%d bytes=%d",
                op_name,
                total_pages,
                total_bytes,
            )
        return results

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        storage_key = self._store_key(key)
        try:
            holder_or_flat = self._get_value(storage_key)
            payload_bytes, keepalive = self._cpu_payload_from_store_value(
                storage_key, holder_or_flat
            )
            if target_location is None:
                if hasattr(self, "mem_pool_host"):
                    target_location = self.mem_pool_host.get_dummy_flat_data_page()
                else:
                    target_location = torch.empty(
                        payload_bytes.numel(), dtype=torch.uint8, device="cpu"
                    )
            result = self._restore_tensor_from_payload(payload_bytes, target_location)
            _ = keepalive
            logger.info(
                "Fluxon get success: key=%s bytes=%d",
                storage_key,
                int(payload_bytes.numel()),
            )
            return result
        except Exception as exc:
            logger.warning("Fluxon get failed for %s: %s", storage_key, exc)
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        if target_locations is None:
            target_locations = [None] * len(keys)
        storage_keys = [self._store_key(key) for key in keys]
        start = time.perf_counter()
        holder_or_flats = self._batch_get_values(storage_keys)
        results: list[torch.Tensor | None] = []
        total_bytes = 0
        for key, storage_key, target_location, holder_or_flat in zip(
            keys, storage_keys, target_locations, holder_or_flats
        ):
            if holder_or_flat is None:
                results.append(None)
                continue
            try:
                payload_bytes, keepalive = self._cpu_payload_from_store_value(
                    storage_key, holder_or_flat
                )
                if target_location is None:
                    if hasattr(self, "mem_pool_host"):
                        target_location = self.mem_pool_host.get_dummy_flat_data_page()
                    else:
                        target_location = torch.empty(
                            payload_bytes.numel(), dtype=torch.uint8, device="cpu"
                        )
                result = self._restore_tensor_from_payload(payload_bytes, target_location)
                _ = keepalive
                total_bytes += int(payload_bytes.numel())
                logger.info(
                    "Fluxon batch_get success: key=%s bytes=%d",
                    storage_key,
                    int(payload_bytes.numel()),
                )
                results.append(result)
            except Exception as exc:
                logger.warning("Fluxon batch_get failed for %s: %s", storage_key, exc)
                results.append(None)
        self._record_prefetch_metrics(len([item for item in results if item is not None]), total_bytes, start)
        return results

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        storage_key = self._store_key(key)
        source = value if value is not None else target_location
        if source is None:
            raise ValueError("Fluxon set requires value or target_location.")
        return self._put_payload(storage_key, source)

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if values is None:
            return False
        value_list = list(values)
        start = time.perf_counter()
        storage_keys = [self._store_key(key) for key in keys]
        put_results = self._batch_put_payloads(storage_keys, value_list)
        ok = all(put_results)
        if ok and value_list:
            total_bytes = sum(
                int(tensor.view(torch.uint8).reshape(-1).numel()) for tensor in value_list
            )
            self._record_backup_metrics(len(storage_keys), total_bytes, start)
        return ok

    def exists(self, key: str) -> bool:
        return self._batch_exists_flags([self._store_key(key)])[0]

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        storage_keys = [self._store_key(key) for key in keys]
        exists_flags = self._batch_exists_flags(storage_keys)
        hit_pages = self._prefix_hit_pages(exists_flags)
        hit_pages = self._pin_existing_prefix(
            storage_keys,
            hit_pages,
            reason="batch_exists_kv",
        )
        self._warm_keys(storage_keys[:hit_pages])
        return hit_pages

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        kv_storage_keys = [self._store_key(key) for key in keys]
        kv_pages = self._prefix_hit_pages(self._batch_exists_flags(kv_storage_keys))
        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages
        extra_pool_keys: dict[Any, List[str]] = {}

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            component_keys = [
                self._store_key(key, transfer.name) for key in keys[:kv_pages]
            ]
            extra_pool_keys[transfer.name] = component_keys
            exists_flags = self._batch_exists_flags(component_keys)

            boundary = 0
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = self._prefix_hit_pages(exists_flags)
            else:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        exists_flags[i]
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break

            if boundary:
                hit_count[transfer.name] = boundary
            final_pages = min(final_pages, boundary)

        final_pages = self._pin_batch_exists_v2_prefix(
            kv_storage_keys, extra_pool_keys, pool_transfers, final_pages
        )
        if final_pages == 0:
            hit_count = {}
        else:
            hit_count = {
                name: min(count, final_pages)
                for name, count in hit_count.items()
                if min(count, final_pages) > 0
            }

        warm_keys = self._batch_exists_v2_storage_keys(
            kv_storage_keys, extra_pool_keys, pool_transfers, final_pages
        )
        self._warm_keys(warm_keys)

        return PoolTransferResult(final_pages, hit_count)

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if not hasattr(self, "mem_pool_host"):
            raise RuntimeError("Fluxon batch_get_v1 requires register_mem_pool_host().")

        page_offsets = self._page_offsets(
            keys, self.mem_pool_host, host_indices, "batch_get_v1"
        )
        if page_offsets is None:
            return [False] * len(keys)

        start = time.perf_counter()
        storage_keys = [self._store_key(key, PoolName.KV) for key in keys]
        results, total_bytes = self._batch_read_pages(
            storage_keys, self.mem_pool_host, page_offsets
        )

        self._record_prefetch_metrics(sum(results), total_bytes, start)
        if any(results):
            logger.info(
                "Fluxon batch_get_v1 success: pages=%d bytes=%d",
                sum(results),
                total_bytes,
            )
        return results

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if not hasattr(self, "mem_pool_host"):
            raise RuntimeError("Fluxon batch_set_v1 requires register_mem_pool_host().")

        page_offsets = self._page_offsets(
            keys, self.mem_pool_host, host_indices, "batch_set_v1"
        )
        if page_offsets is None:
            return [False] * len(keys)

        storage_keys = [self._store_key(key) for key in keys]
        start = time.perf_counter()
        results, total_bytes = self._batch_write_pages(
            storage_keys,
            self.mem_pool_host,
            page_offsets,
            extra_info,
        )
        actual_puts = sum(1 for ok in results if ok)

        self._record_backup_metrics(actual_puts, total_bytes, start)
        if actual_puts > 0:
            logger.info(
                "Fluxon batch_set_v1 success: pages=%d bytes=%d",
                actual_puts,
                total_bytes,
            )
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, "batch_get_v2", extra_info)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, "batch_set_v2", extra_info)

    def clear(self) -> None:
        logger.warning(
            "Fluxon HiCache backend does not implement prefix clear; skipped."
        )

    def close(self) -> None:
        with self._warm_lock:
            self._warm_futures.clear()
            self._warm_inflight.clear()
        executor = getattr(self, "_warm_batch_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if getattr(self, "store", None) is None:
            return
        try:
            staging_pool = self._gpu_direct_staging_pool
            if staging_pool is not None:
                staging_pool.close()
                self._gpu_direct_staging_pool = None
            close_result = self.store.close()
            if hasattr(close_result, "is_ok"):
                if close_result.is_ok():
                    _ = close_result.unwrap()
                else:
                    logger.warning(
                        "Fluxon store close failed: %s",
                        close_result.unwrap_error(),
                    )
        except Exception as exc:
            logger.warning("Fluxon store close raised: %s", exc)
        finally:
            self.store = None

    def get_stats(self):
        if not self.enable_storage_metrics:
            return None
        stats = StorageMetrics(
            prefetch_pgs=self.prefetch_pgs.copy(),
            backup_pgs=self.backup_pgs.copy(),
            prefetch_bandwidth=self.prefetch_bandwidth.copy(),
            backup_bandwidth=self.backup_bandwidth.copy(),
        )
        self._add_observability_delta(stats)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()
        return stats

    def _add_observability_delta(self, stats: StorageMetrics) -> None:
        # Older SGLang versions cannot represent Fluxon's L2/IO observations.
        if not callable(getattr(stats, "add_l2_hit_sample", None)) or not callable(
            getattr(stats, "add_io_sample", None)
        ):
            return
        if getattr(self, "store", None) is None:
            return
        snapshot_fn = getattr(self.store, "observability_snapshot_async", None)
        if snapshot_fn is None:
            return
        try:
            future = snapshot_fn()
            result = future.wait()
            if hasattr(result, "is_ok"):
                if not result.is_ok():
                    logger.warning(
                        "Fluxon observability snapshot failed: %s",
                        result.unwrap_error(),
                    )
                    return
                snapshot = result.unwrap()
            else:
                snapshot = result
            if not isinstance(snapshot, dict):
                logger.warning(
                    "Fluxon observability snapshot returned non-dict: %s",
                    type(snapshot),
                )
                return
        except Exception as exc:
            logger.warning("Fluxon observability snapshot raised: %s", exc)
            return

        previous = self._last_observability_snapshot
        self._last_observability_snapshot = snapshot
        if previous is None:
            return

        def counter_delta(key: str) -> int:
            value = int(snapshot.get(key, 0) or 0)
            prev_value = int(previous.get(key, 0) or 0)
            return max(0, value - prev_value)

        stats.add_l2_hit_sample(
            "local",
            pages=counter_delta("l2_local_hit_pages"),
            num_bytes=counter_delta("l2_local_hit_bytes"),
        )
        stats.add_l2_hit_sample(
            "remote",
            pages=counter_delta("l2_remote_hit_pages"),
            num_bytes=counter_delta("l2_remote_hit_bytes"),
        )

        current_io = snapshot.get("io") or {}
        previous_io = previous.get("io") or {}
        if not isinstance(current_io, dict) or not isinstance(previous_io, dict):
            return

        for key, item in current_io.items():
            if not isinstance(item, dict):
                continue
            prev_item = previous_io.get(key) or {}
            if not isinstance(prev_item, dict):
                prev_item = {}

            try:
                op, locality = str(key).split("_", 1)
            except ValueError:
                continue

            bytes_delta = max(
                0,
                int(item.get("bytes", 0) or 0)
                - int(prev_item.get("bytes", 0) or 0),
            )
            transfer_us_delta = max(
                0,
                int(item.get("transfer_us", 0) or 0)
                - int(prev_item.get("transfer_us", 0) or 0),
            )
            bandwidth_gbps = None
            if bytes_delta > 0 and transfer_us_delta > 0:
                bandwidth_gbps = (bytes_delta * 8.0) / transfer_us_delta / 1000.0
            stats.add_io_sample(op, locality, bytes_delta, bandwidth_gbps)
