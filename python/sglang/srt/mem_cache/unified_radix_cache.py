from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
import time
from array import array
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from queue import Empty
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from sglang.jit_kernel.hicache import (
    can_use_hicache_jit_kernel,
    transfer_hicache_all_layer as jit_transfer_hicache_all_layer,
    transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
)
from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache import memory_pool_host as memory_pool_host_mod
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components import (
    _NUM_COMPONENT_TYPES,
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentData,
    ComponentType,
    EvictLayer,
    FullComponent,
    LRURefreshPhase,
    MambaComponent,
    SWAComponent,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    get_eviction_strategy,
    split_node_hash_value,
)
from sglang.srt.observability.metrics_collector import StorageMetricsCollector
from sglang.srt.platforms import current_platform
from sglang.srt.session.streaming_session import StreamingSession
from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()
_FLUXON_PLAN_BLOB_MAGIC = 0x4658504C414E5631
_FLUXON_GPU_DIRECT_STAGING_SLOT_COUNT = 288
_FLUXON_GPU_DIRECT_STAGING_ENABLED = False
_FLUXON_HOSTLESS_ADMISSION_TOTAL_TOKEN_LIMIT = 234_048
_FLUXON_HOSTLESS_ADMISSION_REMOTE_PAGE_LIMIT = 512
_FLUXON_HOSTLESS_ADMISSION_DEVICE_HEADROOM_TOKENS = 8_192
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import (
        restore_mamba_state_from_fluxon_values,
        restore_mha_pages_from_fluxon_values,
        restore_mla_pages_from_fluxon_values,
        transfer_raw_h2d_batch,
        write_mamba_state_to_fluxon_values,
        write_mha_pages_to_fluxon_values,
        write_mla_pages_to_fluxon_values,
    )
else:

    def transfer_raw_h2d_batch(*args, **kwargs):
        raise RuntimeError(
            "Fluxon hostless raw H2D batch requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )

    def write_mha_pages_to_fluxon_values(*args, **kwargs):
        raise RuntimeError(
            "Fluxon hostless MHA page write requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )

    def restore_mha_pages_from_fluxon_values(*args, **kwargs):
        raise RuntimeError(
            "Fluxon hostless MHA page restore requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )

    def write_mla_pages_to_fluxon_values(*args, **kwargs):
        raise RuntimeError(
            "Fluxon hostless MLA page write requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )

    def restore_mla_pages_from_fluxon_values(*args, **kwargs):
        raise RuntimeError(
            "Fluxon hostless MLA page restore requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )

    def write_mamba_state_to_fluxon_values(*args, **kwargs):
        raise RuntimeError(
            "Fluxon hostless Mamba state write requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )

    def restore_mamba_state_from_fluxon_values(*args, **kwargs):
        raise RuntimeError(
            "Fluxon hostless Mamba state restore requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


class _FluxonRawH2DSubmitState:
    def __init__(self, cache: Any) -> None:
        self._cache = cache
        self._keepalives: list[Any] = []
        self._finalizers: list[Any] = []
        self._has_pending = False

    def retain(self, keepalive: Any) -> None:
        self._keepalives.append(keepalive)

    def add_finalizer(self, finalizer: Any) -> None:
        self._finalizers.append(finalizer)

    def mark_pending(self) -> None:
        self._has_pending = True

    def enqueue(self, dst_ptrs: array, src_ptrs: array, size_bytes: array) -> None:
        if len(dst_ptrs) == 0:
            return
        self._cache._enqueue_raw_h2d_batch(dst_ptrs, src_ptrs, size_bytes)
        self.mark_pending()

    def synchronize(self) -> float:
        if not self._has_pending:
            try:
                for finalizer in self._finalizers:
                    finalizer()
            finally:
                self._finalizers.clear()
                self._keepalives.clear()
            return 0.0
        sync_start = time.perf_counter()
        torch.cuda.current_stream(device=self._cache.device).synchronize()
        self._has_pending = False
        try:
            for finalizer in self._finalizers:
                finalizer()
        finally:
            self._finalizers.clear()
            self._keepalives.clear()
        return (time.perf_counter() - sync_start) * 1000.0


class _FluxonLayerSubmitReadyGuard:
    """Keep consumers from waiting on CUDA events before they are recorded."""

    _MISSING = object()

    def __init__(self, producer_event: Any, num_layers: int) -> None:
        if num_layers <= 0:
            raise ValueError(
                f"Fluxon layer submit guard requires num_layers > 0, got {num_layers}"
            )
        self.producer_event = producer_event
        self.num_layers = num_layers
        self._ready = [threading.Event() for _ in range(num_layers)]
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._installed = False
        self._previous_wait_override: Any = self._MISSING
        self._original_wait = producer_event.wait
        self._guard_wait = self.wait

    def install(self) -> None:
        with self._lock:
            if self._installed:
                raise RuntimeError("Fluxon layer submit guard is already installed")
            event_dict = getattr(self.producer_event, "__dict__", None)
            if event_dict is None:
                raise RuntimeError(
                    "Fluxon layer submit guard requires a mutable producer event"
                )
            self._previous_wait_override = event_dict.get("wait", self._MISSING)
            self.producer_event.wait = self._guard_wait
            self._installed = True

    def mark_submitted(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"Fluxon submitted layer is out of range: {layer_index}/{self.num_layers}"
            )
        self._ready[layer_index].set()

    def fail(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
        for ready in self._ready:
            ready.set()

    def wait(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"Fluxon waited layer is out of range: {layer_index}/{self.num_layers}"
            )
        self._ready[layer_index].wait()
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(
                "Fluxon background DMA submission failed before layer "
                f"{layer_index} became usable"
            ) from error
        self._original_wait(layer_index)

    def uninstall(self) -> None:
        with self._lock:
            if not self._installed:
                return
            event_dict = self.producer_event.__dict__
            if event_dict.get("wait", self._MISSING) is self._guard_wait:
                if self._previous_wait_override is self._MISSING:
                    delattr(self.producer_event, "wait")
                else:
                    self.producer_event.wait = self._previous_wait_override
            self._installed = False


class _FluxonHostlessLayerwiseLoad:
    def __init__(
        self,
        backend: Any,
        plan_ptr: int | None,
        value_ptrs: tuple[int, ...],
        page_indices: torch.Tensor,
        restore_plan: dict[str, Any],
        node_id: int,
        req_id: str,
        token_count: int,
        restored_nodes: list[tuple[Any, torch.Tensor]],
        gpu_staging_lease: Any | None = None,
    ) -> None:
        if plan_ptr is None:
            raise RuntimeError(
                "Fluxon layerwise restore requires one ordered source plan"
            )
        self.backend = backend
        self.plan_ptr: int | None = plan_ptr
        self.gpu_staging_lease = gpu_staging_lease
        self.value_ptrs = value_ptrs
        self.value_ptr_tensor = torch.tensor(value_ptrs, dtype=torch.int64)
        self.page_indices = page_indices
        self.page_index_values = tuple(int(value) for value in page_indices.tolist())
        if len(self.value_ptrs) != len(self.page_index_values):
            raise RuntimeError(
                "Fluxon layerwise restore page/value count mismatch: "
                f"pages={len(self.page_index_values)} values={len(self.value_ptrs)}"
            )
        self.restore_plan = restore_plan
        self.node_id = node_id
        self.req_id = req_id
        self.token_count = token_count
        self.restored_nodes = restored_nodes
        self.submit_keepalive: tuple[Any, ...] | None = None
        self.submit_future: Future | None = None
        self.submit_guard: _FluxonLayerSubmitReadyGuard | None = None
        self.submit_finish_event: Any | None = None
        self.submit_stream: Any | None = None
        self.background_submit_cpu_ms: float | None = None
        self.queued_at = time.perf_counter()

    def release_views(self) -> None:
        if self.plan_ptr is not None:
            plan_ptr = self.plan_ptr
            self.plan_ptr = None
            self.backend.release_views(plan_ptr)
        if self.gpu_staging_lease is not None:
            staging_lease = self.gpu_staging_lease
            self.gpu_staging_lease = None
            staging_lease.release("layerwise_release_views")
        self.submit_keepalive = None


class _FluxonLocalFastPutStartResult:
    def __init__(
        self,
        indices: list[int],
        plan_ptr: int | None,
        filter_ms: float,
        start_ms: float,
        replica_degraded: bool = False,
        replica_error: str | None = None,
    ) -> None:
        self.indices = indices
        self.plan_ptr = plan_ptr
        self.filter_ms = filter_ms
        self.start_ms = start_ms
        self.replica_degraded = replica_degraded
        self.replica_error = replica_error


class _FluxonHostlessFragmentView:
    def __init__(self, fragment_tensors: tuple[torch.Tensor, ...]) -> None:
        self.fragment_tensors = fragment_tensors
        self.fragment_ptrs = [int(t.data_ptr()) for t in fragment_tensors]
        self.fragment_lens = [int(t.numel()) for t in fragment_tensors]
        self.total_bytes = sum(self.fragment_lens)
        self.keepalive = fragment_tensors


class _FluxonHostlessWriteBatch:
    def __init__(
        self,
        node: Any,
        kv_future: Any | None,
        kv_plan_ptr: int | None,
        page_stages: list[Any],
        page_count: int,
        token_count: int,
        total_bytes: int,
        mamba_future: Any | None = None,
        mamba_plan_ptr: int | None = None,
        local_ready_event: Any | None = None,
        local_ready_committed: bool = False,
        write_back: bool = False,
        mamba_bytes: int = 0,
        start_time: float | None = None,
        page_index_prep_ms: float = 0.0,
        kv_local_fast_put_start_ms: float = 0.0,
        kv_write_ms: float = 0.0,
        stream_sync_ms: float = 0.0,
        kv_exist_filter_ms: float = 0.0,
        kv_local_fast_put_commit_ms: float = 0.0,
        mamba_local_fast_put_start_ms: float = 0.0,
        mamba_write_ms: float = 0.0,
        mamba_local_fast_put_commit_ms: float = 0.0,
        mamba_storage_backed: bool = False,
        replica_degraded: bool = False,
        replica_degrade_reasons: list[str] | None = None,
        dedicated_write_stream: bool = False,
    ) -> None:
        self.node = node
        self.kv_future = kv_future
        self.kv_plan_ptr = kv_plan_ptr
        self.page_stages = page_stages
        self.page_count = page_count
        self.token_count = token_count
        self.total_bytes = total_bytes
        self.mamba_future = mamba_future
        self.mamba_plan_ptr = mamba_plan_ptr
        self.local_ready_event = local_ready_event
        self.local_ready_committed = local_ready_committed
        self.write_back = write_back
        self.mamba_bytes = mamba_bytes
        self.page_index_prep_ms = page_index_prep_ms
        self.kv_local_fast_put_start_ms = kv_local_fast_put_start_ms
        self.kv_write_ms = kv_write_ms
        self.stream_sync_ms = stream_sync_ms
        self.kv_exist_filter_ms = kv_exist_filter_ms
        self.kv_local_fast_put_commit_ms = kv_local_fast_put_commit_ms
        self.mamba_local_fast_put_start_ms = mamba_local_fast_put_start_ms
        self.mamba_write_ms = mamba_write_ms
        self.mamba_local_fast_put_commit_ms = mamba_local_fast_put_commit_ms
        self.mamba_storage_backed = mamba_storage_backed
        self.replica_degraded = replica_degraded
        self.replica_degrade_reasons = (
            [] if replica_degrade_reasons is None else replica_degrade_reasons
        )
        self.dedicated_write_stream = dedicated_write_stream
        self.kv_wait_ms = 0.0
        self.mamba_wait_ms = 0.0
        self.start_time = time.perf_counter() if start_time is None else start_time

    def is_waiting(self) -> bool:
        return not self.local_ready_committed

    def clear_keepalives(self) -> None:
        self.page_stages.clear()


class _FluxonHostlessFutureAck:
    def __init__(self, batch: _FluxonHostlessWriteBatch) -> None:
        self.node = batch.node
        self.kv_future = batch.kv_future
        self.mamba_future = batch.mamba_future
        self.page_count = batch.page_count
        self.token_count = batch.token_count
        self.has_mamba = batch.mamba_future is not None
        self.total_bytes = batch.total_bytes
        self.mamba_bytes = batch.mamba_bytes
        self.start_time = time.perf_counter()
        self.keepalive_batch = batch

    def clear_keepalives(self) -> None:
        self.keepalive_batch.clear_keepalives()


class _FluxonHostlessPrefetchOperation:
    def __init__(
        self,
        backend: Any,
        hash_value: list[str],
        kv_handle: Any | None,
        mamba_handle: Any | None,
        mamba_key: str | None,
        completed_tokens: int,
        total_tokens: int,
        anchor_node_id: int,
        has_ready_transfer: bool,
        kv_anchor_node: Any | None = None,
        kv_anchor_lock_params: DecLockRefParams | None = None,
        kv_plan_ptr: int | None = None,
        mamba_plan_ptr: int | None = None,
        mamba_anchor_node_id: int | None = None,
        gpu_staging_lease: Any | None = None,
        gpu_remote_indices: tuple[int, ...] = (),
        admission_total_tokens: int = 0,
        admission_remote_pages: int = 0,
        admission_active: bool = False,
    ) -> None:
        self.backend = backend
        self.hash_value = hash_value
        self.kv_handle = kv_handle
        self.mamba_handle = mamba_handle
        self.mamba_key = mamba_key
        self.completed_tokens = completed_tokens
        self.total_tokens = total_tokens
        self.anchor_node_id = anchor_node_id
        self.kv_anchor_node = kv_anchor_node
        self.kv_anchor_lock_params = kv_anchor_lock_params
        self.kv_plan_ptr = kv_plan_ptr
        self.kv_plan_offset_pages = 0
        self.gpu_staging_lease = gpu_staging_lease
        self.gpu_direct = gpu_staging_lease is not None
        self.gpu_remote_indices = tuple(int(index) for index in gpu_remote_indices)
        self.admission_total_tokens = int(admission_total_tokens)
        self.admission_remote_pages = int(admission_remote_pages)
        self.admission_active = bool(admission_active)
        self.mamba_plan_ptr = mamba_plan_ptr
        self.mamba_anchor_node_id = (
            anchor_node_id if mamba_anchor_node_id is None else mamba_anchor_node_id
        )
        self.has_ready_transfer = has_ready_transfer
        self.start_time = time.monotonic()
        self._terminated = False
        self._finished = True

    def is_terminated(self) -> bool:
        return self._terminated

    def mark_terminate(self) -> None:
        self._terminated = True

    def is_finished(self) -> bool:
        return self._finished


class _FluxonHostlessPageStage(_FluxonHostlessFragmentView):
    def __init__(
        self,
        page_hash: str,
        page_index: int,
        fragment_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        super().__init__(fragment_tensors)
        self.page_hash = page_hash
        self.page_index = page_index


class UnifiedTreeNode:
    counter = 0

    def __init__(self, tree_components: tuple[ComponentType, ...], priority: int = 0):
        self.children = defaultdict(partial(UnifiedTreeNode, tree_components))
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.tree_components = tree_components
        # list indexed by ComponentType (int enum 0..N-1)
        self.component_data: list[ComponentData] = [
            ComponentData() for _ in range(_NUM_COMPONENT_TYPES)
        ]
        self.last_access_time = get_and_increase_time_counter()
        self.creation_time = get_and_increase_time_counter()
        self.hash_value = None
        self.hit_count = 0
        self.priority = priority
        self.lru_prev: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.lru_next: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.id = UnifiedTreeNode.counter
        UnifiedTreeNode.counter += 1

    def component(self, component_type: ComponentType) -> ComponentData:
        return self.component_data[component_type]

    @property
    def backuped(self) -> bool:
        """Tree-level: Full KV recoverable from a lower tier."""
        full_cd = self.component_data[ComponentType.FULL]
        return full_cd.host_value is not None or bool(
            full_cd.metadata.get("storage_backed", False)
        )

    @property
    def evicted(self) -> bool:
        """Tree-level: Full KV not on device (non-root with value=None)."""
        return (
            self.parent is not None
            and self.component_data[ComponentType.FULL].value is None
        )

    def __lt__(self, other: UnifiedTreeNode):
        return self.last_access_time < other.last_access_time

    def get_last_hash_value(self) -> Optional[str]:
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: UnifiedTreeNode) -> list[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value


class UnifiedLRUList:
    def __init__(
        self,
        component_type: ComponentType,
        tree_components: tuple[ComponentType, ...],
        use_host_ptr: bool = False,
    ):
        self.component_type = component_type
        # Pointer slot: host LRU uses offset slots so device/host pointers
        # never collide on the same node.
        self._pt: int = component_type + (_NUM_COMPONENT_TYPES if use_host_ptr else 0)
        self.head = UnifiedTreeNode(tree_components)
        self.tail = UnifiedTreeNode(tree_components)
        self.head.lru_next[self._pt] = self.tail
        self.tail.lru_prev[self._pt] = self.head
        self.cache: dict[int, UnifiedTreeNode] = {}

    def _add_node_after(self, prev_node: UnifiedTreeNode, new_node: UnifiedTreeNode):
        pt = self._pt
        new_node.lru_prev[pt] = prev_node
        new_node.lru_next[pt] = prev_node.lru_next[pt]
        prev_node.lru_next[pt].lru_prev[pt] = new_node
        prev_node.lru_next[pt] = new_node

    def _add_node(self, node: UnifiedTreeNode):
        self._add_node_after(self.head, node)

    def _remove_node(self, node: UnifiedTreeNode):
        pt = self._pt
        node.lru_prev[pt].lru_next[pt] = node.lru_next[pt]
        node.lru_next[pt].lru_prev[pt] = node.lru_prev[pt]
        # Clear self pointers to break reference cycles among evicted nodes.
        node.lru_prev[pt] = None
        node.lru_next[pt] = None

    def insert_mru(self, node: UnifiedTreeNode):
        assert node.id not in self.cache
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        del self.cache[node.id]
        self._remove_node(node)

    def reset_node_mru(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        should_include,
    ):
        prev_node = self.head
        while node != root_node:
            if should_include(node):
                assert node.id in self.cache
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def reset_node_and_window_ancestors_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        window_size: int,
        should_include,
    ):
        prev_node = self.head
        accumulated = 0
        while node != root_node and accumulated < window_size:
            if should_include(node):
                assert node.id in self.cache
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            accumulated += len(node.key)
            node = node.parent

    def in_list(self, node: Optional[UnifiedTreeNode]):
        return node is not None and node.id in self.cache

    def get_prev_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        while x.component_data[ct].lock_ref > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        while x.component_data[ct].lock_ref > 0 or len(x.children) > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_lru_no_lock(self):
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self):
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)


COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
    ComponentType.MAMBA: MambaComponent,
    ComponentType.SWA: SWAComponent,
}

logger = logging.getLogger(__name__)

# External Fluxon `batch_is_exist` is metadata-only on the hostless path. Under
# pressure we have observed positive exists results followed by `get_views`
# misses for the same key, so keep correctness by rewriting pages until the
# backend provides a stronger recoverability guarantee.
_FLUXON_HOSTLESS_USE_EXISTENCE_FILTER = False


class _FluxonHostlessCacheMiss(Exception):
    pass


def _is_fluxon_key_missing_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "keynotfound" in text
        or "key not found" in text
        or "get_views missed key" in text
        or "get_transfer requires all-hit" in text
        or "non-empty transferable prefix" in text
        or "transferable_len" in text
        or "missed page" in text
    )


def _fluxon_key_signature(keys: Sequence[str]) -> tuple[int, str, str, str]:
    if not keys:
        return 0, "", "", ""
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return len(keys), keys[0], keys[-1], digest.hexdigest()[:16]


class UnifiedRadixCache(KVCacheEventMixin, BasePrefixCache):
    def __init__(
        self,
        params: CacheInitParams,
    ):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.is_eagle = params.is_eagle
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.kv_event_queue = []
        self.eviction_policy = params.eviction_policy.lower()
        self.eviction_strategy = get_eviction_strategy(self.eviction_policy)

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()
        self._enable_metrics_flag = params.enable_metrics
        self.enable_storage_metrics = False
        self.storage_metrics_collector: Optional[StorageMetricsCollector] = None
        self.extra_metric_labels = None

        assert params.tree_components is not None
        self.tree_components = tuple(params.tree_components)
        component_registry = COMPONENT_REGISTRY
        if params.component_registry_override:
            component_registry = {
                **COMPONENT_REGISTRY,
                **params.component_registry_override,
            }
        self.components: dict[ComponentType, TreeComponent] = {
            ct: component_registry[ct](self, params) for ct in self.tree_components
        }
        self._components_tuple: tuple[TreeComponent, ...] = tuple(
            self.components.values()
        )
        self.sidecar_pool_specs: list[SidecarPoolSpec] = []

        # Streaming session: embedded StreamingSession with self as inner.
        # Always on -- zero overhead when no streaming session is open (the
        # try_* entries short-circuit on non-streaming reqs / real TreeNodes).
        # Dispatch methods below pre-check conditions so the session's
        # internal fall-through to self.inner.xxx never fires -- no recursion.
        self.session = StreamingSession(inner=self)

        self.tp_group = params.tp_cache_group
        self.tp_world_size = (
            1
            if self.tp_group is None
            else torch.distributed.get_world_size(group=self.tp_group)
        )

        # HiCache D↔H defaults (overridden by init_hicache)
        self.cache_controller = None
        self.write_through_threshold = 256
        self.prefetch_stop_policy = "best_effort"
        self.prefetch_threshold = 256
        self.prefetch_timeout_base = 1.0
        self.prefetch_timeout_per_page = 0.25
        self.hicache_storage_pass_prefix_keys = False
        self._fluxon_hostless_plan_cache = None
        self._fluxon_hostless_mamba_plan_cache = None
        # Fluxon enables its fixed r133/r134 path in init_hicache(), once the
        # selected storage backend is known. Keep ordinary HiCache backends
        # independent of the Fluxon CUDA worker and policy.
        self._fluxon_hostless_layer_batch_dma_enabled = False
        self._fluxon_hostless_background_dma_submit_enabled = False
        self._fluxon_hostless_dma_submit_executor = None
        self._fluxon_hostless_dma_submit_thread_state = threading.local()
        self._fluxon_hostless_cuda_device_id: int | None = None
        self._fluxon_hostless_layer_batch_dma_batches = 0
        self._fluxon_hostless_layer_batch_dma_pages = 0
        self._fluxon_hostless_page_index_validate_every_n = 0
        self._fluxon_hostless_page_index_call_count = 0
        self._fluxon_hostless_eviction_write_stream_enabled = False

        self.reset()
        logger.info(f"Init Unified RadixTree with components {self.tree_components}")

    def reset(self) -> None:
        self._reset_full()

    def _enable_fluxon_hostless_runtime(self) -> None:
        """Enable the single validated E44 r133/r134 CUDA runtime path."""
        if not (_is_cuda or _is_hip):
            raise RuntimeError(
                "Fluxon hostless runtime requires a CUDA or ROCm SGLang build"
            )
        if self._fluxon_hostless_dma_submit_executor is not None:
            raise RuntimeError(
                "Fluxon hostless runtime was initialized more than once"
            )

        # Resolve an index-less device on the scheduler thread. A fresh Python
        # worker otherwise starts on device 0, which is wrong for nonzero TP
        # ranks. The worker creates and owns its submission stream.
        self._fluxon_hostless_cuda_device_id = self._cuda_device_index()
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fluxon-h2d-submit",
        )
        self._fluxon_hostless_dma_submit_executor = executor
        try:
            executor.submit(self._fluxon_hostless_background_dma_stream).result()
        except Exception:
            self._fluxon_hostless_dma_submit_executor = None
            self._fluxon_hostless_cuda_device_id = None
            executor.shutdown(wait=True, cancel_futures=True)
            raise

        self._fluxon_hostless_layer_batch_dma_enabled = True
        self._fluxon_hostless_background_dma_submit_enabled = True
        self._fluxon_hostless_eviction_write_stream_enabled = True

    def _reset_full(self) -> None:
        """Full reset: destroy entire tree and all state."""
        if hasattr(self, "_fluxon_hostless_observation_counters"):
            self._log_fluxon_hostless_observation_snapshot("reset")
        if hasattr(self, "fluxon_hostless_load_queue"):
            self._clear_fluxon_hostless_layerwise_loads()
        if hasattr(self, "fluxon_hostless_ready_prefetch"):
            self._clear_fluxon_hostless_ready_prefetch()
        if hasattr(self, "ongoing_prefetch"):
            for info in list(self.ongoing_prefetch.values()):
                operation = info[3]
                if isinstance(operation, _FluxonHostlessPrefetchOperation):
                    self._cancel_fluxon_hostless_prefetch_operation(
                        operation,
                        "reset_ongoing_prefetch",
                    )
        if hasattr(self, "ongoing_fluxon_hostless_backup"):
            self._flush_pending_fluxon_hostless_backups()
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.root_node.priority = -sys.maxsize
        self.root_node.key = RadixKey(array("q"), None)
        self.root_node.component_data[BASE_COMPONENT_TYPE].value = []
        self.root_node.hash_value = []
        for ct in self.tree_components:
            self.root_node.component_data[ct].lock_ref = 1
        self.component_evictable_size_ = {ct: 0 for ct in self.tree_components}
        self.component_protected_size_ = {ct: 0 for ct in self.tree_components}

        self.lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components) for ct in self.tree_components
        }
        self.session.slots.clear()

        self.evictable_device_leaves: set[UnifiedTreeNode] = set()
        self.evictable_host_leaves: set[UnifiedTreeNode] = set()
        self.host_lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components, use_host_ptr=True)
            for ct in self.tree_components
        }
        self.ongoing_write_through: dict[
            int, tuple[UnifiedTreeNode, Optional[DecLockRefParams]]
        ] = {}
        self.ongoing_load_back: dict[int, tuple[UnifiedTreeNode, DecLockRefParams]] = {}
        self.fluxon_hostless_load_queue: list[_FluxonHostlessLayerwiseLoad] = []
        self.ongoing_fluxon_hostless_layerwise_load: dict[
            int, _FluxonHostlessLayerwiseLoad
        ] = {}
        self.ongoing_fluxon_hostless_backup: dict[int, _FluxonHostlessWriteBatch] = {}
        self.ongoing_fluxon_hostless_acks: dict[
            tuple[int, int], _FluxonHostlessFutureAck
        ] = {}
        self.enable_storage = False
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.ongoing_prefetch: dict = {}
        self.fluxon_hostless_ready_prefetch: dict[str, _FluxonHostlessPrefetchOperation] = {}
        self.ongoing_backup: dict = {}
        self._fluxon_hostless_request_observations: dict[str, dict[str, Any]] = {}
        self._fluxon_hostless_observation_counters: dict[str, int] = defaultdict(int)
        self._fluxon_hostless_eviction_observation_stack: list[dict[str, Any]] = []
        self._fluxon_hostless_anchor_locks_active = 0
        self._fluxon_hostless_anchor_locks_acquired = 0
        self._fluxon_hostless_anchor_locks_released = 0
        self._fluxon_hostless_anchor_locks_high_watermark = 0
        self._fluxon_hostless_admission_total_tokens_occupied = 0
        self._fluxon_hostless_admission_remote_pages_occupied = 0
        self._fluxon_hostless_admission_active = 0
        self._fluxon_hostless_admission_acquired = 0
        self._fluxon_hostless_admission_released = 0
        self._fluxon_hostless_admission_total_rejected = 0
        self._fluxon_hostless_admission_remote_rejected = 0
        self._fluxon_hostless_admission_state_rejected = 0
        self._fluxon_hostless_admission_device_headroom_rejected = 0
        self._fluxon_hostless_admission_source_mismatches = 0
        self._fluxon_hostless_admission_total_tokens_high_watermark = 0
        self._fluxon_hostless_admission_remote_pages_high_watermark = 0
        self._fluxon_hostless_admission_active_high_watermark = 0

        if self.cache_controller is not None:
            self.cache_controller.reset()
            self.cache_controller.mem_pool_host.clear()
            self.enable_storage = self.cache_controller.enable_storage

        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
        )
        self._record_all_cleared_event()

    def init_hicache(self, server_args: ServerArgs, params: CacheInitParams) -> None:
        """Initialize HiCache infrastructure."""
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            attach_hybrid_pool_to_unified_cache,
        )

        # Direct IO layout fixup (must happen before pool creation)
        if server_args.hicache_io_backend == "direct":
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, "
                    "switching to page first direct layout"
                )

        self.load_cache_event = threading.Event()
        self.sidecar_pool_specs.clear()
        self.extra_metric_labels = server_args.extra_metric_labels

        # Parse storage config once, share with assembler and tree
        storage_backend = server_args.hicache_storage_backend
        storage_extra_config = None
        storage_prefetch_threshold = 256
        prefetch_timeout_base = 1.0
        prefetch_timeout_per_ki_token = 0.25
        hicache_storage_pass_prefix_keys = False
        if storage_backend is not None:
            (
                storage_extra_config,
                storage_prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = HybridCacheController.parse_storage_backend_extra_config(
                server_args.hicache_storage_backend_extra_config,
                storage_backend,
            )
            if str(storage_backend).lower() == "fluxon":
                if server_args.hicache_write_policy != "write_back":
                    raise ValueError(
                        "Fluxon E44 r134 requires --hicache-write-policy "
                        "write_back"
                    )
                # Keep the validated E44 r134 Fluxon policy fixed. The only
                # user-provided adapter setting is the Fluxon YAML path.
                storage_prefetch_threshold = 64
                hicache_storage_pass_prefix_keys = True
                self._enable_fluxon_hostless_runtime()
                logger.warning(
                    "Fluxon hostless restore uses one arbitrary-address H2D DMA "
                    "batch per model layer"
                )

        attach_hybrid_pool_to_unified_cache(
            self,
            params,
            server_args,
            load_cache_event=self.load_cache_event,
            attn_cp_group=params.attn_cp_cache_group,
            attn_tp_group=params.attn_tp_cache_group,
            storage_backend=storage_backend,
            storage_extra_config=storage_extra_config,
            storage_prefetch_threshold=storage_prefetch_threshold,
        )

        # State initialization
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 256
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        if storage_backend is not None:
            self._apply_storage_runtime_config(
                storage_backend=storage_backend,
                prefetch_threshold=storage_prefetch_threshold,
                prefetch_timeout_base=prefetch_timeout_base,
                prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
                enable_storage=self.cache_controller.enable_storage,
                enable_storage_metrics=self._enable_metrics_flag,
                extra_metric_labels=self.extra_metric_labels,
            )
            if str(storage_backend).lower() == "fluxon":
                if _FLUXON_GPU_DIRECT_STAGING_ENABLED:
                    self._configure_fluxon_gpu_direct_staging()
                else:
                    logger.warning(
                        "Fluxon GPU-direct staging disabled: mode=cpu_h2d_only"
                    )

    def register_sidecar_pool(self, spec: SidecarPoolSpec) -> None:
        self.sidecar_pool_specs.append(spec)

    def _is_fluxon_hostless_full_mode(self) -> bool:
        component_set = set(self.tree_components)
        if BASE_COMPONENT_TYPE not in component_set:
            return False
        if component_set - {BASE_COMPONENT_TYPE, ComponentType.MAMBA}:
            return False
        if not self.enable_storage or self.cache_controller is None:
            return False
        if self.cache_controller.storage_backend_type != "fluxon":
            return False
        return True

    def _is_fluxon_hostless_mamba_mode(self) -> bool:
        return (
            self._is_fluxon_hostless_full_mode()
            and ComponentType.MAMBA in self.components
        )

    def _record_fluxon_hostless_source_ready_event(
        self, node: UnifiedTreeNode
    ) -> None:
        """Fence immutable radix-node data after its device value is installed."""
        if not getattr(
            self, "_fluxon_hostless_eviction_write_stream_enabled", False
        ):
            return
        if not self._is_fluxon_hostless_full_mode() or not _is_cuda:
            return
        full_cd = node.component_data[BASE_COMPONENT_TYPE]
        if full_cd.value is None:
            return
        try:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device=self.device))
            full_cd.metadata["fluxon_hostless_source_ready_event"] = event
        except Exception as err:
            full_cd.metadata.pop("fluxon_hostless_source_ready_event", None)
            logger.warning(
                "Fluxon source-ready event record failed; eviction write will "
                "use the current stream: node=%d error=%s",
                node.id,
                err,
            )

    def _fluxon_hostless_eviction_stream_guard(
        self, node: UnifiedTreeNode, write_back: bool
    ) -> tuple[Any | None, Any | None]:
        """Return the Fluxon stream only for immutable source-ready pages."""
        if not getattr(
            self, "_fluxon_hostless_eviction_write_stream_enabled", False
        ):
            return None, None
        if not write_back:
            return None, None
        full_cd = node.component_data[BASE_COMPONENT_TYPE]
        if full_cd.value is None:
            return None, None
        source_ready_event = full_cd.metadata.get(
            "fluxon_hostless_source_ready_event"
        )
        write_stream = getattr(self.cache_controller, "write_stream", None)
        if source_ready_event is None or write_stream is None:
            return None, None
        return write_stream, source_ready_event

    def _fluxon_backend(self):
        if self.cache_controller is None:
            return None
        return self.cache_controller.storage_backend

    def _configure_fluxon_gpu_direct_staging(self) -> None:
        if not _is_cuda or not self._is_fluxon_hostless_full_mode():
            return
        backend = self._fluxon_backend()
        if backend is None:
            raise RuntimeError("Fluxon GPU staging requires a storage backend")
        plan = self._fluxon_hostless_plan()
        value_len = int(plan["total_bytes"])
        backend.configure_gpu_direct_staging(
            value_len=value_len,
            slot_count=_FLUXON_GPU_DIRECT_STAGING_SLOT_COUNT,
            device_id=self._cuda_device_index(),
        )
        logger.warning(
            "Fluxon GPU-direct staging enabled: slots=%d value_len=%d bytes=%d device=%d",
            _FLUXON_GPU_DIRECT_STAGING_SLOT_COUNT,
            value_len,
            _FLUXON_GPU_DIRECT_STAGING_SLOT_COUNT * value_len,
            self._cuda_device_index(),
        )

    def _clear_fluxon_hostless_layerwise_loads(self) -> None:
        pending = list(getattr(self, "fluxon_hostless_load_queue", []))
        inflight_by_node = getattr(
            self, "ongoing_fluxon_hostless_layerwise_load", {}
        )
        inflight = list(inflight_by_node.values())
        futures = {
            operation.submit_future
            for operation in pending + inflight
            if operation.submit_future is not None
        }
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                logger.warning(
                    "Fluxon background DMA submit failed during clear: %s", exc
                )
        finish_events = {
            operation.submit_finish_event
            for operation in pending + inflight
            if operation.submit_finish_event is not None
        }
        for finish_event in finish_events:
            try:
                finish_event.synchronize()
            except Exception as exc:
                logger.warning(
                    "Fluxon layerwise finish event failed during clear: %s", exc
                )
        guards = {
            operation.submit_guard
            for operation in pending + inflight
            if operation.submit_guard is not None
        }
        for guard in guards:
            guard.uninstall()
        seen: set[int] = set()
        for operation in pending + inflight:
            operation_id = id(operation)
            if operation_id in seen:
                continue
            seen.add(operation_id)
            self._abort_fluxon_hostless_layerwise_load(operation)
        self.fluxon_hostless_load_queue.clear()
        inflight_by_node.clear()

    def _abort_fluxon_hostless_layerwise_load(
        self,
        operation: _FluxonHostlessLayerwiseLoad,
    ) -> None:
        self.ongoing_fluxon_hostless_layerwise_load.pop(operation.node_id, None)
        try:
            operation.release_views()
        except Exception as exc:
            logger.warning(
                "Fluxon layerwise restore view release failed during abort: "
                "node=%d error=%s",
                operation.node_id,
                exc,
            )
        for node, device_indices in operation.restored_nodes:
            cd = node.component_data[BASE_COMPONENT_TYPE]
            if cd.value is not None:
                cd.value = None
                self.component_evictable_size_[BASE_COMPONENT_TYPE] -= len(
                    device_indices
                )
                self._update_evictable_leaf_sets(node)
                if node.parent is not None:
                    self._update_evictable_leaf_sets(node.parent)
            self.token_to_kv_pool_allocator.free(device_indices)
        lock_entry = self.ongoing_load_back.pop(operation.node_id, None)
        if lock_entry is not None:
            node, lock_params = lock_entry
            self.dec_lock_ref(node, lock_params)
        if operation.req_id in self._fluxon_hostless_request_observations:
            self._finish_fluxon_hostless_request_observation(
                operation.req_id,
                "load_back_dma_aborted",
                restore_complete_ms=(
                    time.perf_counter() - operation.queued_at
                )
                * 1000.0,
            )

    def _build_fluxon_hostless_restore_kernel_plans(
        self,
        operations: list[_FluxonHostlessLayerwiseLoad],
    ) -> dict[str, Any]:
        if not operations:
            raise RuntimeError("Fluxon layerwise restore kernel batch is empty")
        plan = operations[0].restore_plan
        layer_num = int(plan["layer_num"])
        for operation in operations:
            if operation.plan_ptr is None:
                raise RuntimeError(
                    "Fluxon layerwise restore lost its source: "
                    f"node={operation.node_id}"
                )
            if operation.restore_plan["cache_key"] != plan["cache_key"]:
                raise RuntimeError(
                    "Fluxon layerwise restore batch mixes different KV pool layouts"
                )
        value_ptrs = torch.cat(
            [operation.value_ptr_tensor for operation in operations]
        ).to(device="cpu", dtype=torch.int64)
        page_count = sum(
            int(operation.page_indices.numel()) for operation in operations
        )
        if value_ptrs.device.type != "cpu":
            raise RuntimeError("Fluxon kernel value pointers must be a CPU tensor")
        if page_count != value_ptrs.numel():
            raise RuntimeError(
                "Fluxon layerwise restore kernel page/value mismatch: "
                f"pages={page_count} values={value_ptrs.numel()}"
            )
        value_ptrs = value_ptrs.contiguous()
        layer_offsets = torch.arange(layer_num, dtype=torch.int64).reshape(-1, 1)
        values = value_ptrs.reshape(1, -1)
        if plan["is_mla"]:
            page_bytes = int(plan["page_bytes"])
            src_ptrs = values + layer_offsets * page_bytes
            # The existing one-component restore kernel accepts a plan blob.
            # Build one pinned blob per layer so its source offset is explicit.
            plan_blobs = torch.empty(
                (layer_num, 1, page_count + 2),
                dtype=torch.int64,
                pin_memory=True,
            )
            plan_blobs[:, :, 0] = _FLUXON_PLAN_BLOB_MAGIC
            plan_blobs[:, :, 1] = page_count
            plan_blobs[:, 0, 2:] = src_ptrs
            return {"plan": plan, "plan_blobs": plan_blobs}
        k_page_bytes = int(plan["k_page_bytes"])
        v_page_bytes = int(plan["v_page_bytes"])
        k_src_ptrs = values + layer_offsets * k_page_bytes
        v_src_ptrs = (
            values
            + layer_num * k_page_bytes
            + layer_offsets * v_page_bytes
        )
        plan_blobs = torch.empty(
            (layer_num, 2, page_count + 2),
            dtype=torch.int64,
            pin_memory=True,
        )
        plan_blobs[:, :, 0] = _FLUXON_PLAN_BLOB_MAGIC
        plan_blobs[:, :, 1] = page_count
        # Reuse the one-component kernel independently for K and V. This keeps
        # every layer event precise without issuing thousands of memcpy calls.
        plan_blobs[:, 0, 2:] = k_src_ptrs
        plan_blobs[:, 1, 2:] = v_src_ptrs
        return {"plan": plan, "plan_blobs": plan_blobs}

    def _enqueue_fluxon_hostless_restore_kernels(
        self,
        operations: list[_FluxonHostlessLayerwiseLoad],
        kernel_plans: dict[str, Any],
        producer_event: Any,
    ) -> tuple[Any, ...]:
        plan = kernel_plans["plan"]
        plan_blobs = kernel_plans["plan_blobs"]
        layer_num = int(plan["layer_num"])
        device_id = self._cuda_device_index()
        device = torch.device("cuda", device_id)
        page_indices = torch.cat(
            [operation.page_indices for operation in operations]
        ).to(device=device, dtype=torch.int64).contiguous()
        if int(page_indices.numel()) + 2 != int(plan_blobs.shape[2]):
            raise RuntimeError(
                "Fluxon layerwise restore kernel plan/page mismatch: "
                f"pages={page_indices.numel()} plan_values={plan_blobs.shape[2] - 2}"
            )

        if plan["is_mla"]:
            page_bytes = int(plan["page_bytes"])
            layer_ptrs = plan["layer_ptrs"].to(
                device=device, dtype=torch.int64
            ).contiguous()
            for layer_id in range(layer_num):
                restore_mla_pages_from_fluxon_values(
                    plan_blobs[layer_id, 0].data_ptr(),
                    page_indices,
                    layer_ptrs[layer_id : layer_id + 1],
                    page_bytes,
                    device_id,
                )
                producer_event.complete(layer_id)
            return page_indices, plan_blobs, layer_ptrs

        k_page_bytes = int(plan["k_page_bytes"])
        v_page_bytes = int(plan["v_page_bytes"])
        k_layer_ptrs = plan["k_layer_ptrs"].to(
            device=device, dtype=torch.int64
        ).contiguous()
        v_layer_ptrs = plan["v_layer_ptrs"].to(
            device=device, dtype=torch.int64
        ).contiguous()
        for layer_id in range(layer_num):
            restore_mla_pages_from_fluxon_values(
                plan_blobs[layer_id, 0].data_ptr(),
                page_indices,
                k_layer_ptrs[layer_id : layer_id + 1],
                k_page_bytes,
                device_id,
            )
            restore_mla_pages_from_fluxon_values(
                plan_blobs[layer_id, 1].data_ptr(),
                page_indices,
                v_layer_ptrs[layer_id : layer_id + 1],
                v_page_bytes,
                device_id,
            )
            producer_event.complete(layer_id)
        return (
            page_indices,
            plan_blobs,
            k_layer_ptrs,
            v_layer_ptrs,
        )

    def _build_fluxon_hostless_layer_batch_dma_plan(
        self,
        operations: list[_FluxonHostlessLayerwiseLoad],
    ) -> dict[str, Any]:
        if not operations:
            raise RuntimeError("Fluxon layer-batched DMA restore batch is empty")
        if any(operation.gpu_staging_lease is not None for operation in operations):
            raise RuntimeError(
                "Fluxon layer-batched H2D DMA cannot consume GPU staging sources"
            )
        plan = operations[0].restore_plan
        for operation in operations:
            if operation.restore_plan["cache_key"] != plan["cache_key"]:
                raise RuntimeError(
                    "Fluxon layer-batched DMA mixes different KV pool layouts"
                )

        value_ptrs = torch.cat(
            [operation.value_ptr_tensor for operation in operations]
        ).to(device="cpu", dtype=torch.int64).contiguous()
        page_indices = torch.tensor(
            [
                page_index
                for operation in operations
                for page_index in operation.page_index_values
            ],
            dtype=torch.int64,
        )
        if value_ptrs.numel() != page_indices.numel():
            raise RuntimeError(
                "Fluxon layer-batched DMA page/value mismatch: "
                f"pages={page_indices.numel()} values={value_ptrs.numel()}"
            )

        layer_num = int(plan["layer_num"])
        layer_offsets = torch.arange(layer_num, dtype=torch.int64).reshape(-1, 1)
        values = value_ptrs.reshape(1, -1)
        if plan["is_mla"]:
            page_bytes = int(plan["page_bytes"])
            src_ptrs = values + layer_offsets * page_bytes
            dst_bases = torch.tensor(
                plan["layer_ptr_values"], dtype=torch.int64
            ).reshape(-1, 1)
            dst_ptrs = dst_bases + page_indices.reshape(1, -1) * page_bytes
            size_bytes = torch.full_like(src_ptrs, page_bytes)
        else:
            k_page_bytes = int(plan["k_page_bytes"])
            v_page_bytes = int(plan["v_page_bytes"])
            k_src_ptrs = values + layer_offsets * k_page_bytes
            v_src_ptrs = (
                values
                + layer_num * k_page_bytes
                + layer_offsets * v_page_bytes
            )
            k_dst_bases = torch.tensor(
                plan["k_layer_ptr_values"], dtype=torch.int64
            ).reshape(-1, 1)
            v_dst_bases = torch.tensor(
                plan["v_layer_ptr_values"], dtype=torch.int64
            ).reshape(-1, 1)
            k_dst_ptrs = (
                k_dst_bases + page_indices.reshape(1, -1) * k_page_bytes
            )
            v_dst_ptrs = (
                v_dst_bases + page_indices.reshape(1, -1) * v_page_bytes
            )
            src_ptrs = torch.cat((k_src_ptrs, v_src_ptrs), dim=1)
            dst_ptrs = torch.cat((k_dst_ptrs, v_dst_ptrs), dim=1)
            size_bytes = torch.cat(
                (
                    torch.full_like(k_src_ptrs, k_page_bytes),
                    torch.full_like(v_src_ptrs, v_page_bytes),
                ),
                dim=1,
            )

        return {
            "plan": plan,
            "src_ptrs": src_ptrs.contiguous(),
            "dst_ptrs": dst_ptrs.contiguous(),
            "size_bytes": size_bytes.contiguous(),
            "page_count": int(page_indices.numel()),
        }

    def _enqueue_fluxon_hostless_layer_batch_dma(
        self,
        dma_plan: dict[str, Any],
        producer_event: Any,
    ) -> tuple[Any, ...]:
        plan = dma_plan["plan"]
        src_ptrs = dma_plan["src_ptrs"]
        dst_ptrs = dma_plan["dst_ptrs"]
        size_bytes = dma_plan["size_bytes"]
        layer_num = int(plan["layer_num"])
        if src_ptrs.shape != dst_ptrs.shape or src_ptrs.shape != size_bytes.shape:
            raise RuntimeError("Fluxon layer-batched DMA descriptor shape mismatch")
        if int(src_ptrs.shape[0]) != layer_num:
            raise RuntimeError(
                "Fluxon layer-batched DMA layer mismatch: "
                f"descriptors={src_ptrs.shape[0]} layers={layer_num}"
            )

        device_id = self._cuda_device_index()
        for layer_id in range(layer_num):
            transfer_raw_h2d_batch(
                dst_ptrs[layer_id],
                src_ptrs[layer_id],
                size_bytes[layer_id],
                device_id,
            )
            producer_event.complete(layer_id)
        return src_ptrs, dst_ptrs, size_bytes

    def _submit_fluxon_hostless_layer_batch_dma_background(
        self,
        dma_plan: dict[str, Any],
        producer_event: Any,
        submit_guard: _FluxonLayerSubmitReadyGuard,
        operations: list[_FluxonHostlessLayerwiseLoad],
    ) -> None:
        """Submit per-layer copies away from the scheduler's Python thread."""
        cc = self.cache_controller
        if cc is None:
            error = RuntimeError(
                "Fluxon background DMA submission lost its cache controller"
            )
            submit_guard.fail(error)
            raise error
        plan = dma_plan["plan"]
        src_ptrs = dma_plan["src_ptrs"]
        dst_ptrs = dma_plan["dst_ptrs"]
        size_bytes = dma_plan["size_bytes"]
        layer_num = int(plan["layer_num"])
        device_id = self._cuda_device_index()
        submit_stream = self._fluxon_hostless_background_dma_stream()
        for operation in operations:
            operation.submit_stream = submit_stream
        submitted_layers = 0
        submit_start = time.perf_counter()
        submit_error: BaseException | None = None
        try:
            torch.cuda.set_device(device_id)
            with torch.cuda.device(device_id), torch.cuda.stream(submit_stream):
                producer_event.start_event.wait(submit_stream)
                for layer_id in range(layer_num):
                    transfer_raw_h2d_batch(
                        dst_ptrs[layer_id],
                        src_ptrs[layer_id],
                        size_bytes[layer_id],
                        device_id,
                    )
                    # complete() must return before the CPU guard opens. CUDA
                    # treats a wait on a never-recorded event as already done.
                    producer_event.complete(layer_id)
                    submitted_layers = layer_id + 1
                    submit_guard.mark_submitted(layer_id)
        except BaseException as exc:
            submit_error = exc
            submit_guard.fail(exc)
            # Leave the counter reusable and make finish_event observable even
            # when a descriptor/API call fails. Consumers still see the guard
            # error and cannot use the incomplete destination pages.
            try:
                torch.cuda.set_device(device_id)
                with torch.cuda.device(device_id), torch.cuda.stream(submit_stream):
                    for layer_id in range(submitted_layers, layer_num):
                        producer_event.complete(layer_id)
            except BaseException as completion_exc:
                logger.exception(
                    "Fluxon background DMA failed to close remaining layer events: "
                    "submitted=%d layers=%d original_error=%s completion_error=%s",
                    submitted_layers,
                    layer_num,
                    exc,
                    completion_exc,
                )
            raise
        finally:
            elapsed_ms = (time.perf_counter() - submit_start) * 1000.0
            for operation in operations:
                operation.background_submit_cpu_ms = elapsed_ms
            if submit_error is None:
                # Every CUDA event is now recorded, so the normal nonblocking
                # wait_event path is safe again even while copies are pending.
                submit_guard.uninstall()
            logger.info(
                "Fluxon background layer DMA submit complete: operations=%d "
                "tokens=%d layers=%d pages=%d submitted_layers=%d "
                "submit_cpu_ms=%.3f error=%s",
                len(operations),
                sum(operation.token_count for operation in operations),
                layer_num,
                int(dma_plan["page_count"]),
                submitted_layers,
                elapsed_ms,
                submit_error,
            )

    def _fluxon_hostless_background_dma_stream(self) -> Any:
        state = self._fluxon_hostless_dma_submit_thread_state
        device_id = self._cuda_device_index()
        stream = getattr(state, "stream", None)
        stream_device_id = getattr(state, "device_id", None)
        if stream is not None and stream_device_id == device_id:
            return stream
        torch.cuda.set_device(device_id)
        with torch.cuda.device(device_id):
            stream = torch.cuda.Stream(device=device_id)
            # Force thread-local CUDA state to be established during startup.
            with torch.cuda.stream(stream):
                torch.cuda.current_stream(device=device_id)
        state.stream = stream
        state.device_id = device_id
        logger.info(
            "Fluxon background DMA worker initialized: device=%d stream=%s",
            device_id,
            getattr(stream, "cuda_stream", "unknown"),
        )
        return stream

    def _start_fluxon_hostless_layerwise_loads(self) -> int:
        if not self.fluxon_hostless_load_queue:
            return -1
        cc = self.cache_controller
        if cc is None:
            raise RuntimeError("Fluxon layerwise restore requires a cache controller")

        operations = list(self.fluxon_hostless_load_queue)
        layer_num = int(operations[0].restore_plan["layer_num"])
        if layer_num <= 0:
            raise RuntimeError(
                f"Fluxon layerwise restore requires layer_num > 0, got {layer_num}"
            )
        if int(cc.layer_done_counter.num_layers) != layer_num:
            raise RuntimeError(
                "Fluxon layerwise restore/controller layer mismatch: "
                f"restore={layer_num} controller={cc.layer_done_counter.num_layers}"
            )
        for operation in operations:
            if int(operation.restore_plan["layer_num"]) != layer_num:
                raise RuntimeError(
                    "Fluxon layerwise restore batch contains inconsistent layer counts"
                )

        producer_id = cc.layer_done_counter.update_producer()
        producer_event = cc.layer_done_counter.events[producer_id]
        self.fluxon_hostless_load_queue.clear()
        producer_event.start_event.record()
        submit_guard: _FluxonLayerSubmitReadyGuard | None = None
        submit_future: Future | None = None
        for operation in operations:
            operation.submit_finish_event = producer_event.finish_event
        try:
            descriptor_start = time.perf_counter()
            has_gpu_staging = any(
                operation.gpu_staging_lease is not None
                for operation in operations
            )
            if (
                self._fluxon_hostless_layer_batch_dma_enabled
                and not has_gpu_staging
            ):
                dma_plan = self._build_fluxon_hostless_layer_batch_dma_plan(
                    operations
                )
                kernel_plans = None
                transport = "layer_batch_dma"
            else:
                dma_plan = None
                kernel_plans = self._build_fluxon_hostless_restore_kernel_plans(
                    operations
                )
                transport = (
                    "gpu_direct_d2d_kernel"
                    if has_gpu_staging
                    else "kernel"
                )
            descriptor_cpu_ms = (time.perf_counter() - descriptor_start) * 1000.0
            submit_start = time.perf_counter()
            if (
                dma_plan is not None
                and self._fluxon_hostless_background_dma_submit_enabled
            ):
                executor = self._fluxon_hostless_dma_submit_executor
                if executor is None:
                    raise RuntimeError(
                        "Fluxon background DMA submission executor is unavailable"
                    )
                submit_guard = _FluxonLayerSubmitReadyGuard(
                    producer_event,
                    layer_num,
                )
                submit_guard.install()
                submit_keepalive = (
                    dma_plan["src_ptrs"],
                    dma_plan["dst_ptrs"],
                    dma_plan["size_bytes"],
                )
                for operation in operations:
                    operation.submit_keepalive = submit_keepalive
                    operation.submit_guard = submit_guard
                submit_future = executor.submit(
                    self._submit_fluxon_hostless_layer_batch_dma_background,
                    dma_plan,
                    producer_event,
                    submit_guard,
                    operations,
                )
                for operation in operations:
                    operation.submit_future = submit_future
                transport = "layer_batch_dma_background"
            else:
                with torch.cuda.stream(cc.load_stream):
                    producer_event.start_event.wait(cc.load_stream)
                    if dma_plan is not None:
                        submit_keepalive = (
                            self._enqueue_fluxon_hostless_layer_batch_dma(
                                dma_plan,
                                producer_event,
                            )
                        )
                    else:
                        submit_keepalive = (
                            self._enqueue_fluxon_hostless_restore_kernels(
                                operations,
                                kernel_plans,
                                producer_event,
                            )
                        )
            # Pinned plan blobs and merged CUDA indices must outlive every
            # layer kernel. release_views() clears them after finish_event.
            for operation in operations:
                operation.submit_keepalive = submit_keepalive
            if dma_plan is not None:
                self._fluxon_hostless_layer_batch_dma_batches += 1
                self._fluxon_hostless_layer_batch_dma_pages += int(
                    dma_plan["page_count"]
                )
        except Exception as exc:
            if submit_guard is not None:
                submit_guard.fail(exc)
            if submit_future is not None:
                try:
                    submit_future.result()
                except Exception:
                    pass
            producer_event.finish_event.synchronize()
            with torch.cuda.stream(cc.load_stream):
                for layer_id in range(layer_num):
                    producer_event.complete(layer_id)
            cc.load_stream.synchronize()
            if submit_guard is not None:
                submit_guard.uninstall()
            for operation in operations:
                self._abort_fluxon_hostless_layerwise_load(operation)
            raise

        node_ids = [operation.node_id for operation in operations]
        cc.ack_load_queue.append(
            (producer_event.start_event, producer_event.finish_event, node_ids)
        )
        logger.info(
            "Fluxon layerwise restore submitted: transport=%s producer=%d "
            "operations=%d tokens=%d layers=%d pages=%d "
            "dma_batches_total=%d dma_pages_total=%d descriptor_cpu_ms=%.3f "
            "dispatch_cpu_ms=%.3f background=%s",
            transport,
            producer_id,
            len(operations),
            sum(operation.token_count for operation in operations),
            layer_num,
            sum(int(operation.page_indices.numel()) for operation in operations),
            self._fluxon_hostless_layer_batch_dma_batches,
            self._fluxon_hostless_layer_batch_dma_pages,
            descriptor_cpu_ms,
            (time.perf_counter() - submit_start) * 1000.0,
            submit_future is not None,
        )
        return producer_id

    def _cancel_fluxon_hostless_get_start_handle(
        self,
        backend: Any,
        handle: Any | None,
        caller: str,
        node_id: int | None = None,
        gpu_direct: bool = False,
        plan_only: bool = False,
    ) -> None:
        if handle is None:
            return
        if bool(getattr(handle, "closed", False)):
            return
        try:
            if plan_only:
                backend.cancel_get_plan(handle)
            elif gpu_direct:
                backend.cancel_get_transfer_gpu(handle)
            else:
                backend.cancel_get_transfer(handle)
        except Exception as exc:
            logger.warning(
                "Fluxon hostless get_start handle cancel failed: "
                "node=%s caller=%s gpu_direct=%s plan_only=%s error=%s",
                str(node_id),
                caller,
                gpu_direct,
                plan_only,
                exc,
            )

    def _release_fluxon_gpu_staging_lease(
        self,
        staging_lease: Any | None,
        caller: str,
    ) -> None:
        if staging_lease is None:
            return
        try:
            staging_lease.release(caller)
        except Exception as exc:
            logger.warning(
                "Fluxon GPU staging release failed: caller=%s error=%s",
                caller,
                exc,
            )

    def _acquire_fluxon_hostless_anchor_lock(
        self,
        operation: _FluxonHostlessPrefetchOperation,
        anchor_node: UnifiedTreeNode,
    ) -> None:
        if anchor_node is self.root_node:
            return
        if (
            operation.kv_anchor_node is not None
            or operation.kv_anchor_lock_params is not None
        ):
            raise RuntimeError("Fluxon hostless anchor lock already installed")
        lock_result = self.inc_lock_ref(anchor_node)
        operation.kv_anchor_node = anchor_node
        operation.kv_anchor_lock_params = lock_result.to_dec_params()
        self._fluxon_hostless_anchor_locks_active += 1
        self._fluxon_hostless_anchor_locks_acquired += 1
        self._fluxon_hostless_anchor_locks_high_watermark = max(
            self._fluxon_hostless_anchor_locks_high_watermark,
            self._fluxon_hostless_anchor_locks_active,
        )

    def _release_fluxon_hostless_anchor_lock(
        self,
        operation: _FluxonHostlessPrefetchOperation,
    ) -> None:
        anchor_node = operation.kv_anchor_node
        lock_params = operation.kv_anchor_lock_params
        operation.kv_anchor_node = None
        operation.kv_anchor_lock_params = None
        if anchor_node is None:
            if lock_params is not None:
                raise RuntimeError(
                    "Fluxon hostless anchor lock params exist without an anchor"
                )
            return
        if lock_params is None:
            raise RuntimeError("Fluxon hostless anchor lock is missing release params")
        self.dec_lock_ref(anchor_node, lock_params)
        self._fluxon_hostless_anchor_locks_active -= 1
        self._fluxon_hostless_anchor_locks_released += 1
        if self._fluxon_hostless_anchor_locks_active < 0:
            raise RuntimeError("Fluxon hostless anchor lock count became negative")

    def _try_acquire_fluxon_hostless_prefetch_admission(
        self,
        req_id: str,
        total_tokens: int,
        local_remote_pages: int,
    ) -> dict[str, Any]:
        """Admit one metadata-only plan by holder and real remote-source debt."""
        total_tokens = int(total_tokens)
        local_remote_pages = int(local_remote_pages)
        if total_tokens <= 0:
            raise ValueError(
                "Fluxon hostless admission requires positive total_tokens"
            )
        if local_remote_pages < 0:
            raise ValueError(
                "Fluxon hostless admission remote pages cannot be negative"
            )
        if self.cache_controller is None:
            raise RuntimeError("Fluxon hostless admission lost its cache controller")

        if self.supports_swa():
            device_available_local = int(
                self.token_to_kv_pool_allocator.full_available_size()
            )
        else:
            device_available_local = int(
                self.token_to_kv_pool_allocator.available_size()
            )
        device_evictable_local = int(
            self.component_evictable_size_.get(BASE_COMPONENT_TYPE, 0)
        )
        device_reclaimable_local = max(
            0,
            device_available_local + device_evictable_local,
        )

        total_before = int(
            self._fluxon_hostless_admission_total_tokens_occupied
        )
        remote_before = int(
            self._fluxon_hostless_admission_remote_pages_occupied
        )
        controller_before = int(self.cache_controller.prefetch_tokens_occupied)
        state_min = torch.tensor(
            [
                total_before,
                remote_before,
                controller_before,
                total_tokens,
                local_remote_pages,
                device_reclaimable_local,
            ],
            dtype=torch.int64,
        )
        state_max = state_min.clone()
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                state_min,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            torch.distributed.all_reduce(
                state_max,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group,
            )

        occupied_consistent = all(
            int(state_min[index].item()) == int(state_max[index].item())
            for index in range(3)
        )
        requested_total_consistent = (
            int(state_min[3].item()) == int(state_max[3].item())
        )
        source_min_pages = int(state_min[4].item())
        remote_pages = int(state_max[4].item())
        device_reclaimable_min = int(state_min[5].item())
        device_reclaimable_max = int(state_max[5].item())
        device_prefetch_budget = max(
            0,
            device_reclaimable_min
            - _FLUXON_HOSTLESS_ADMISSION_DEVICE_HEADROOM_TOKENS,
        )
        source_mismatch = source_min_pages != remote_pages
        if source_mismatch:
            self._fluxon_hostless_admission_source_mismatches += 1

        reason = "admitted"
        if (
            not occupied_consistent
            or not requested_total_consistent
            or total_before != controller_before
        ):
            reason = "tp_state_mismatch"
            self._fluxon_hostless_admission_state_rejected += 1
        elif (
            total_before + total_tokens
            > _FLUXON_HOSTLESS_ADMISSION_TOTAL_TOKEN_LIMIT
        ):
            reason = "total_holder_limit"
            self._fluxon_hostless_admission_total_rejected += 1
        elif (
            remote_before + remote_pages
            > _FLUXON_HOSTLESS_ADMISSION_REMOTE_PAGE_LIMIT
        ):
            reason = "remote_source_limit"
            self._fluxon_hostless_admission_remote_rejected += 1
        elif total_before + total_tokens > device_prefetch_budget:
            reason = "device_headroom"
            self._fluxon_hostless_admission_device_headroom_rejected += 1

        admitted = reason == "admitted"
        if admitted:
            self._fluxon_hostless_admission_total_tokens_occupied += total_tokens
            self._fluxon_hostless_admission_remote_pages_occupied += remote_pages
            self._fluxon_hostless_admission_active += 1
            self._fluxon_hostless_admission_acquired += 1
            self.cache_controller.prefetch_tokens_occupied += total_tokens
            self._fluxon_hostless_admission_total_tokens_high_watermark = max(
                self._fluxon_hostless_admission_total_tokens_high_watermark,
                self._fluxon_hostless_admission_total_tokens_occupied,
            )
            self._fluxon_hostless_admission_remote_pages_high_watermark = max(
                self._fluxon_hostless_admission_remote_pages_high_watermark,
                self._fluxon_hostless_admission_remote_pages_occupied,
            )
            self._fluxon_hostless_admission_active_high_watermark = max(
                self._fluxon_hostless_admission_active_high_watermark,
                self._fluxon_hostless_admission_active,
            )

        total_after = int(
            self._fluxon_hostless_admission_total_tokens_occupied
        )
        remote_after = int(
            self._fluxon_hostless_admission_remote_pages_occupied
        )
        logger.info(
            "Fluxon hostless source admission acquire: req=%s tp_rank=%d "
            "admitted=%s reason=%s total_tokens=%d remote_pages=%d "
            "source_min_pages=%d source_max_pages=%d source_mismatch=%s "
            "total_before=%d total_after=%d total_limit=%d "
            "remote_before=%d remote_after=%d remote_limit_pages=%d "
            "device_reclaimable_min=%d device_reclaimable_max=%d "
            "device_prefetch_budget=%d device_headroom_tokens=%d "
            "active=%d acquired=%d released=%d",
            req_id,
            self._fluxon_hostless_tp_rank(),
            admitted,
            reason,
            total_tokens,
            remote_pages,
            source_min_pages,
            remote_pages,
            source_mismatch,
            total_before,
            total_after,
            _FLUXON_HOSTLESS_ADMISSION_TOTAL_TOKEN_LIMIT,
            remote_before,
            remote_after,
            _FLUXON_HOSTLESS_ADMISSION_REMOTE_PAGE_LIMIT,
            device_reclaimable_min,
            device_reclaimable_max,
            device_prefetch_budget,
            _FLUXON_HOSTLESS_ADMISSION_DEVICE_HEADROOM_TOKENS,
            self._fluxon_hostless_admission_active,
            self._fluxon_hostless_admission_acquired,
            self._fluxon_hostless_admission_released,
        )
        return {
            "admitted": admitted,
            "reason": reason,
            "total_tokens": total_tokens,
            "remote_pages": remote_pages,
            "source_min_pages": source_min_pages,
            "source_max_pages": remote_pages,
            "total_before": total_before,
            "total_after": total_after,
            "remote_before": remote_before,
            "remote_after": remote_after,
            "device_reclaimable_min": device_reclaimable_min,
            "device_reclaimable_max": device_reclaimable_max,
            "device_prefetch_budget": device_prefetch_budget,
        }

    def _release_fluxon_hostless_prefetch_admission_values(
        self,
        total_tokens: int,
        remote_pages: int,
        caller: str,
    ) -> None:
        total_tokens = int(total_tokens)
        remote_pages = int(remote_pages)
        total_before = int(
            self._fluxon_hostless_admission_total_tokens_occupied
        )
        remote_before = int(
            self._fluxon_hostless_admission_remote_pages_occupied
        )
        if (
            total_tokens <= 0
            or remote_pages < 0
            or total_tokens > total_before
            or remote_pages > remote_before
            or self._fluxon_hostless_admission_active <= 0
        ):
            raise RuntimeError(
                "Fluxon hostless admission release invariant failed: "
                f"caller={caller} total={total_tokens}/{total_before} "
                f"remote={remote_pages}/{remote_before} "
                f"active={self._fluxon_hostless_admission_active}"
            )
        if self.cache_controller is None:
            raise RuntimeError("Fluxon hostless admission release lost its controller")
        controller_before = int(self.cache_controller.prefetch_tokens_occupied)
        if total_tokens > controller_before:
            raise RuntimeError(
                "Fluxon hostless controller admission debt became negative: "
                f"caller={caller} release={total_tokens} occupied={controller_before}"
            )

        self._fluxon_hostless_admission_total_tokens_occupied -= total_tokens
        self._fluxon_hostless_admission_remote_pages_occupied -= remote_pages
        self._fluxon_hostless_admission_active -= 1
        self._fluxon_hostless_admission_released += 1
        self.cache_controller.prefetch_tokens_occupied -= total_tokens
        logger.info(
            "Fluxon hostless source admission release: caller=%s tp_rank=%d "
            "total_tokens=%d remote_pages=%d total_before=%d total_after=%d "
            "remote_before=%d remote_after=%d active=%d acquired=%d released=%d",
            caller,
            self._fluxon_hostless_tp_rank(),
            total_tokens,
            remote_pages,
            total_before,
            self._fluxon_hostless_admission_total_tokens_occupied,
            remote_before,
            self._fluxon_hostless_admission_remote_pages_occupied,
            self._fluxon_hostless_admission_active,
            self._fluxon_hostless_admission_acquired,
            self._fluxon_hostless_admission_released,
        )
        if self._fluxon_hostless_admission_active == 0:
            self._log_fluxon_hostless_admission_snapshot(f"{caller}:drained")

    def _release_fluxon_hostless_prefetch_admission(
        self,
        operation: _FluxonHostlessPrefetchOperation,
        caller: str,
    ) -> None:
        if not bool(getattr(operation, "admission_active", False)):
            return
        total_tokens = int(operation.admission_total_tokens)
        remote_pages = int(operation.admission_remote_pages)
        self._release_fluxon_hostless_prefetch_admission_values(
            total_tokens,
            remote_pages,
            caller,
        )
        operation.admission_active = False
        operation.admission_total_tokens = 0
        operation.admission_remote_pages = 0

    def _cancel_fluxon_hostless_prefetch_operation(
        self,
        operation: _FluxonHostlessPrefetchOperation,
        caller: str,
    ) -> None:
        self._release_fluxon_hostless_prefetch_admission(operation, caller)
        self._release_fluxon_hostless_anchor_lock(operation)
        if operation.kv_plan_ptr is not None:
            operation.backend.release_views(operation.kv_plan_ptr)
            operation.kv_plan_ptr = None
        if operation.mamba_plan_ptr is not None:
            operation.backend.release_views(operation.mamba_plan_ptr)
            operation.mamba_plan_ptr = None
        self._cancel_fluxon_hostless_get_start_handle(
            operation.backend,
            operation.kv_handle,
            caller,
            gpu_direct=operation.gpu_direct,
        )
        operation.kv_handle = None
        self._cancel_fluxon_hostless_get_start_handle(
            operation.backend,
            operation.mamba_handle,
            caller,
        )
        operation.mamba_handle = None
        if operation.gpu_staging_lease is not None:
            staging_lease = operation.gpu_staging_lease
            operation.gpu_staging_lease = None
            self._release_fluxon_gpu_staging_lease(staging_lease, caller)

    def _clear_fluxon_hostless_ready_prefetch(self) -> None:
        for operation in list(self.fluxon_hostless_ready_prefetch.values()):
            self._cancel_fluxon_hostless_prefetch_operation(
                operation,
                "reset_ready_prefetch",
            )
        self.fluxon_hostless_ready_prefetch.clear()

    def _fluxon_hostless_tp_rank(self) -> int:
        if self.cache_controller is None:
            return -1
        return int(getattr(self.cache_controller, "tp_rank", -1))

    def _fluxon_hostless_tp_plan_execute_commit(
        self,
        *,
        local_succeeded: bool,
        gpu_direct: bool,
    ) -> tuple[bool, int, int]:
        """Commit Plan execution only when every TP rank reached one mode."""
        execute_state = torch.tensor(
            [
                int(local_succeeded),
                int(local_succeeded and gpu_direct),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                execute_state,
                op=torch.distributed.ReduceOp.SUM,
                group=self.tp_group,
            )
        succeeded_rank_count = int(execute_state[0].item())
        gpu_direct_rank_count = int(execute_state[1].item())
        mode_consistent = gpu_direct_rank_count in (0, self.tp_world_size)
        committed = (
            succeeded_rank_count == self.tp_world_size and mode_consistent
        )
        return committed, succeeded_rank_count, gpu_direct_rank_count

    def _observe_fluxon_hostless_request(
        self,
        req_id: str,
        **fields: Any,
    ) -> dict[str, Any]:
        observation = self._fluxon_hostless_request_observations.get(req_id)
        if observation is None:
            observation = {
                "created_at": time.monotonic(),
                "prefetch_decision": "entered",
                "anchor_node_id": -1,
                "prefetch_input_tokens": 0,
                "host_hit_tokens": 0,
                "requested_pages": 0,
                "gpu_direct_admission_reason": "not_observed",
                "gpu_direct_requested_pages": 0,
                "gpu_direct_selected": 0,
                "gpu_direct_capacity_slots": 0,
                "gpu_direct_free_slots_before": 0,
                "gpu_direct_live_slots_before": 0,
                "gpu_direct_active_leases_before": 0,
                "gpu_direct_free_slots_after": 0,
                "gpu_direct_live_slots_after": 0,
                "gpu_direct_active_leases_after": 0,
                "gpu_direct_high_watermark_slots": 0,
                "gpu_direct_tp_min_pages": 0,
                "gpu_direct_tp_max_pages": 0,
                "initial_transferable_pages": 0,
                "final_transferable_pages": 0,
                "tp_min_pages": 0,
                "tp_max_pages": 0,
                "retry_count": 0,
                "initial_start_ms": 0.0,
                "retry_start_ms": 0.0,
                "get_transfer_ms": 0.0,
                "ready_pages": 0,
                "ready_bytes": 0,
                "ready_wait_ms": 0.0,
                "consumed_bytes": 0,
                "restore_complete_ms": 0.0,
                "scheduler_enqueue_queue_position": -1,
                "scheduler_enqueue_queue_length": 0,
                "scheduler_enqueue_pending_tokens": 0,
                "scheduler_enqueue_uncached_tokens": 0,
                "scheduler_scan_count": 0,
                "scheduler_first_scan_age_ms": 0.0,
                "scheduler_consume_queue_position": -1,
                "scheduler_consume_queue_length": 0,
                "scheduler_consume_pending_tokens": 0,
                "scheduler_consume_uncached_tokens": 0,
                "plan_ready_age_ms": 0.0,
                "gpu_reserve_attempt_age_ms": 0.0,
                "gpu_reserve_age_ms": 0.0,
                "gpu_execute_return_age_ms": 0.0,
                "gpu_backend_handle": -1,
                "transfer_consume_start_age_ms": 0.0,
                "rdma_start_age_ms": 0.0,
                "rdma_terminal_age_ms": 0.0,
                "rdma_transfer_wall_ms": 0.0,
                "rdma_terminal_before_consume": 0,
                "rdma_terminal_to_consume_ms": 0.0,
                "rdma_finish_wait_ms": 0.0,
                "load_back_consume_start_age_ms": 0.0,
                "restore_queued_age_ms": 0.0,
                "restore_complete_age_ms": 0.0,
                "staging_release_age_ms": 0.0,
                "lineage_plan_unix_ns": 0,
                "lineage_plan_handle": -1,
                "lineage_start_depth_pages": 0,
                "lineage_cpu_plan_pages": 0,
                "lineage_gpu_plan_pages": 0,
                "lineage_materialization": "none",
                "lineage_key_ids": (),
                "lineage_sources": "",
            }
            self._fluxon_hostless_request_observations[req_id] = observation
        observation.update(fields)
        return observation

    def _fluxon_hostless_observation_age_ms(self, req_id: str) -> float:
        observation = self._fluxon_hostless_request_observations.get(req_id)
        if observation is None:
            return 0.0
        return (time.monotonic() - observation["created_at"]) * 1000.0

    @staticmethod
    def _fluxon_kv_lineage_key_id(key: str) -> str:
        """Return a stable compact ID without writing the full storage key."""
        return hashlib.blake2b(
            str(key).encode("utf-8"),
            digest_size=8,
            person=b"fluxon-r60",
        ).hexdigest()

    def _emit_fluxon_kv_lineage(
        self,
        req_id: str,
        terminal: str,
        observation: dict[str, Any],
    ) -> None:
        key_ids = tuple(observation.get("lineage_key_ids", ()))
        sources = str(observation.get("lineage_sources", ""))
        if not key_ids:
            return
        if len(key_ids) != len(sources):
            logger.error(
                "Fluxon KV lineage shape mismatch: req=%s keys=%d sources=%d",
                req_id,
                len(key_ids),
                len(sources),
            )
            return
        payload = {
            "schema": "e44_r60_kv_lineage_v1",
            "req": req_id,
            "tp_rank": self._fluxon_hostless_tp_rank(),
            "terminal": terminal,
            "plan_unix_ns": int(observation.get("lineage_plan_unix_ns", 0)),
            "terminal_unix_ns": time.time_ns(),
            "plan_handle": int(observation.get("lineage_plan_handle", -1)),
            "anchor_node": int(observation.get("anchor_node_id", -1)),
            "start_depth_pages": int(
                observation.get("lineage_start_depth_pages", 0)
            ),
            "requested_pages": int(observation.get("requested_pages", 0)),
            "transferable_pages": len(key_ids),
            "cpu_plan_pages": int(
                observation.get("lineage_cpu_plan_pages", 0)
            ),
            "gpu_plan_pages": int(
                observation.get("lineage_gpu_plan_pages", 0)
            ),
            "materialization": str(
                observation.get("lineage_materialization", "none")
            ),
            "gpu_direct_selected": int(
                observation.get("gpu_direct_selected", 0)
            ),
            "key_ids": key_ids,
            "sources": sources,
        }
        logger.info(
            "Fluxon KV lineage: %s",
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )

    def observe_fluxon_prefetch_scheduler_state(
        self,
        req_id: str,
        *,
        phase: str,
        queue_position: int,
        queue_length: int,
        pending_tokens: int,
        uncached_tokens: int,
    ) -> None:
        if not self._is_fluxon_hostless_full_mode():
            return
        observation = self._observe_fluxon_hostless_request(req_id)
        if phase == "enqueue":
            observation.update(
                scheduler_enqueue_queue_position=int(queue_position),
                scheduler_enqueue_queue_length=int(queue_length),
                scheduler_enqueue_pending_tokens=int(pending_tokens),
                scheduler_enqueue_uncached_tokens=int(uncached_tokens),
            )
            return
        if phase != "consume":
            raise ValueError(f"unknown Fluxon scheduler observation phase: {phase}")
        observation["scheduler_scan_count"] = int(
            observation.get("scheduler_scan_count", 0)
        ) + 1
        if float(observation.get("scheduler_first_scan_age_ms", 0.0)) == 0.0:
            observation["scheduler_first_scan_age_ms"] = (
                self._fluxon_hostless_observation_age_ms(req_id)
            )
        observation.update(
            scheduler_consume_queue_position=int(queue_position),
            scheduler_consume_queue_length=int(queue_length),
            scheduler_consume_pending_tokens=int(pending_tokens),
            scheduler_consume_uncached_tokens=int(uncached_tokens),
        )

    def _finish_fluxon_hostless_request_observation(
        self,
        req_id: str,
        terminal: str,
        **fields: Any,
    ) -> None:
        observation = self._fluxon_hostless_request_observations.pop(req_id, None)
        if observation is None:
            observation = {
                "created_at": time.monotonic(),
                "prefetch_decision": "untracked",
            }
        observation.update(fields)
        total_ms = (time.monotonic() - observation["created_at"]) * 1000.0
        self._emit_fluxon_kv_lineage(req_id, terminal, observation)
        decision = str(observation.get("prefetch_decision", "unknown"))
        counters = self._fluxon_hostless_observation_counters
        counters["terminal_events"] += 1
        counters[f"terminal.{terminal}"] += 1
        counters[f"decision.{decision}"] += 1
        gpu_direct_admission_reason = str(
            observation.get("gpu_direct_admission_reason", "not_observed")
        )
        counters[
            f"gpu_direct_admission.{gpu_direct_admission_reason}"
        ] += 1
        counters["gpu_direct_requested_pages"] += int(
            observation.get("gpu_direct_requested_pages", 0)
        )
        counters["gpu_direct_selected"] += int(
            observation.get("gpu_direct_selected", 0)
        )
        counters["gpu_direct_high_watermark_slots"] = max(
            counters["gpu_direct_high_watermark_slots"],
            int(observation.get("gpu_direct_high_watermark_slots", 0)),
        )
        counters["host_hit_tokens"] += int(observation.get("host_hit_tokens", 0))
        counters["ready_pages"] += int(observation.get("ready_pages", 0))
        counters["consumed_pages"] += int(observation.get("consumed_pages", 0))
        counters["consumed_tokens"] += int(observation.get("consumed_tokens", 0))
        counters["consumed_bytes"] += int(observation.get("consumed_bytes", 0))
        counters["evict_requested_tokens"] += int(
            observation.get("evict_requested_tokens", 0)
        )
        counters["evict_actual_tokens"] += int(
            observation.get("evict_actual_tokens", 0)
        )
        logger.info(
            "Fluxon hostless request lifecycle: req=%s tp_rank=%d "
            "terminal=%s decision=%s anchor_node=%s prefetch_input_tokens=%d "
            "host_hit_tokens=%d requested_pages=%d initial_transferable_pages=%d "
            "final_transferable_pages=%d tp_min_pages=%d tp_max_pages=%d "
            "gpu_direct_admission=%s gpu_direct_requested_pages=%d "
            "gpu_direct_selected=%d gpu_direct_capacity_slots=%d "
            "gpu_direct_free_slots_before=%d gpu_direct_live_slots_before=%d "
            "gpu_direct_active_leases_before=%d gpu_direct_free_slots_after=%d "
            "gpu_direct_live_slots_after=%d gpu_direct_active_leases_after=%d "
            "gpu_direct_high_watermark_slots=%d "
            "gpu_direct_tp_min_pages=%d gpu_direct_tp_max_pages=%d "
            "retry_count=%d ready_pages=%d ready_bytes=%d "
            "consumed_pages=%d consumed_tokens=%d consumed_bytes=%d "
            "initial_start_ms=%.3f "
            "retry_start_ms=%.3f get_transfer_ms=%.3f ready_wait_ms=%.3f "
            "evict_requested_tokens=%d evict_actual_tokens=%d "
            "evict_candidate_tokens=%d evict_already_backed_tokens=%d "
            "evict_after_writeback_tokens=%d evict_unbacked_drop_tokens=%d "
            "evict_new_writebacks=%d evict_pending_writebacks=%d "
            "evict_write_backup_ms=%.3f evict_write_wait_ms=%.3f "
            "evict_free_group_ms=%.3f evict_total_ms=%.3f "
            "restore_complete_ms=%.3f total_ms=%.3f",
            req_id,
            self._fluxon_hostless_tp_rank(),
            terminal,
            decision,
            observation.get("anchor_node_id", -1),
            int(observation.get("prefetch_input_tokens", 0)),
            int(observation.get("host_hit_tokens", 0)),
            int(observation.get("requested_pages", 0)),
            int(observation.get("initial_transferable_pages", 0)),
            int(observation.get("final_transferable_pages", 0)),
            int(observation.get("tp_min_pages", 0)),
            int(observation.get("tp_max_pages", 0)),
            gpu_direct_admission_reason,
            int(observation.get("gpu_direct_requested_pages", 0)),
            int(observation.get("gpu_direct_selected", 0)),
            int(observation.get("gpu_direct_capacity_slots", 0)),
            int(observation.get("gpu_direct_free_slots_before", 0)),
            int(observation.get("gpu_direct_live_slots_before", 0)),
            int(observation.get("gpu_direct_active_leases_before", 0)),
            int(observation.get("gpu_direct_free_slots_after", 0)),
            int(observation.get("gpu_direct_live_slots_after", 0)),
            int(observation.get("gpu_direct_active_leases_after", 0)),
            int(observation.get("gpu_direct_high_watermark_slots", 0)),
            int(observation.get("gpu_direct_tp_min_pages", 0)),
            int(observation.get("gpu_direct_tp_max_pages", 0)),
            int(observation.get("retry_count", 0)),
            int(observation.get("ready_pages", 0)),
            int(observation.get("ready_bytes", 0)),
            int(observation.get("consumed_pages", 0)),
            int(observation.get("consumed_tokens", 0)),
            int(observation.get("consumed_bytes", 0)),
            float(observation.get("initial_start_ms", 0.0)),
            float(observation.get("retry_start_ms", 0.0)),
            float(observation.get("get_transfer_ms", 0.0)),
            float(observation.get("ready_wait_ms", 0.0)),
            int(observation.get("evict_requested_tokens", 0)),
            int(observation.get("evict_actual_tokens", 0)),
            int(observation.get("evict_candidate_tokens", 0)),
            int(observation.get("evict_already_backed_tokens", 0)),
            int(observation.get("evict_after_writeback_tokens", 0)),
            int(observation.get("evict_unbacked_drop_tokens", 0)),
            int(observation.get("evict_new_writebacks", 0)),
            int(observation.get("evict_pending_writebacks", 0)),
            float(observation.get("evict_write_backup_ms", 0.0)),
            float(observation.get("evict_write_wait_ms", 0.0)),
            float(observation.get("evict_free_group_ms", 0.0)),
            float(observation.get("evict_total_ms", 0.0)),
            float(observation.get("restore_complete_ms", 0.0)),
            total_ms,
        )
        logger.info(
            "Fluxon prefetch timeline: req=%s tp_rank=%d terminal=%s "
            "enqueue_pos=%d enqueue_len=%d enqueue_pending_tokens=%d "
            "enqueue_uncached_tokens=%d scheduler_scan_count=%d "
            "first_scan_age_ms=%.3f consume_pos=%d consume_len=%d "
            "consume_pending_tokens=%d consume_uncached_tokens=%d "
            "plan_ready_age_ms=%.3f reserve_attempt_age_ms=%.3f "
            "reserve_age_ms=%.3f execute_return_age_ms=%.3f "
            "backend_handle=%d transfer_consume_start_age_ms=%.3f "
            "rdma_start_age_ms=%.3f rdma_terminal_age_ms=%.3f "
            "rdma_transfer_wall_ms=%.3f terminal_before_consume=%d "
            "terminal_to_consume_ms=%.3f rdma_finish_wait_ms=%.3f "
            "load_back_consume_start_age_ms=%.3f restore_queued_age_ms=%.3f "
            "restore_complete_age_ms=%.3f staging_release_age_ms=%.3f "
            "total_ms=%.3f",
            req_id,
            self._fluxon_hostless_tp_rank(),
            terminal,
            int(observation.get("scheduler_enqueue_queue_position", -1)),
            int(observation.get("scheduler_enqueue_queue_length", 0)),
            int(observation.get("scheduler_enqueue_pending_tokens", 0)),
            int(observation.get("scheduler_enqueue_uncached_tokens", 0)),
            int(observation.get("scheduler_scan_count", 0)),
            float(observation.get("scheduler_first_scan_age_ms", 0.0)),
            int(observation.get("scheduler_consume_queue_position", -1)),
            int(observation.get("scheduler_consume_queue_length", 0)),
            int(observation.get("scheduler_consume_pending_tokens", 0)),
            int(observation.get("scheduler_consume_uncached_tokens", 0)),
            float(observation.get("plan_ready_age_ms", 0.0)),
            float(observation.get("gpu_reserve_attempt_age_ms", 0.0)),
            float(observation.get("gpu_reserve_age_ms", 0.0)),
            float(observation.get("gpu_execute_return_age_ms", 0.0)),
            int(observation.get("gpu_backend_handle", -1)),
            float(observation.get("transfer_consume_start_age_ms", 0.0)),
            float(observation.get("rdma_start_age_ms", 0.0)),
            float(observation.get("rdma_terminal_age_ms", 0.0)),
            float(observation.get("rdma_transfer_wall_ms", 0.0)),
            int(observation.get("rdma_terminal_before_consume", 0)),
            float(observation.get("rdma_terminal_to_consume_ms", 0.0)),
            float(observation.get("rdma_finish_wait_ms", 0.0)),
            float(observation.get("load_back_consume_start_age_ms", 0.0)),
            float(observation.get("restore_queued_age_ms", 0.0)),
            float(observation.get("restore_complete_age_ms", 0.0)),
            float(observation.get("staging_release_age_ms", 0.0)),
            total_ms,
        )

    def _log_fluxon_hostless_admission_snapshot(self, caller: str) -> None:
        logger.info(
            "Fluxon hostless source admission Snapshot: caller=%s tp_rank=%d "
            "active=%d acquired=%d released=%d total_occupied=%d "
            "remote_pages_occupied=%d total_high_watermark=%d "
            "remote_pages_high_watermark=%d active_high_watermark=%d "
            "total_rejected=%d remote_rejected=%d state_rejected=%d "
            "device_headroom_rejected=%d source_mismatches=%d "
            "total_limit=%d remote_limit_pages=%d device_headroom_tokens=%d",
            caller,
            self._fluxon_hostless_tp_rank(),
            int(getattr(self, "_fluxon_hostless_admission_active", 0)),
            int(getattr(self, "_fluxon_hostless_admission_acquired", 0)),
            int(getattr(self, "_fluxon_hostless_admission_released", 0)),
            int(
                getattr(
                    self,
                    "_fluxon_hostless_admission_total_tokens_occupied",
                    0,
                )
            ),
            int(
                getattr(
                    self,
                    "_fluxon_hostless_admission_remote_pages_occupied",
                    0,
                )
            ),
            int(
                getattr(
                    self,
                    "_fluxon_hostless_admission_total_tokens_high_watermark",
                    0,
                )
            ),
            int(
                getattr(
                    self,
                    "_fluxon_hostless_admission_remote_pages_high_watermark",
                    0,
                )
            ),
            int(
                getattr(
                    self,
                    "_fluxon_hostless_admission_active_high_watermark",
                    0,
                )
            ),
            int(
                getattr(
                    self,
                    "_fluxon_hostless_admission_total_rejected",
                    0,
                )
            ),
            int(
                getattr(
                    self,
                    "_fluxon_hostless_admission_remote_rejected",
                    0,
                )
            ),
            int(
                getattr(
                    self,
                    "_fluxon_hostless_admission_state_rejected",
                    0,
                )
            ),
            int(
                getattr(
                    self,
                    "_fluxon_hostless_admission_device_headroom_rejected",
                    0,
                )
            ),
            int(
                getattr(
                    self,
                    "_fluxon_hostless_admission_source_mismatches",
                    0,
                )
            ),
            _FLUXON_HOSTLESS_ADMISSION_TOTAL_TOKEN_LIMIT,
            _FLUXON_HOSTLESS_ADMISSION_REMOTE_PAGE_LIMIT,
            _FLUXON_HOSTLESS_ADMISSION_DEVICE_HEADROOM_TOKENS,
        )

    def _log_fluxon_hostless_observation_snapshot(self, caller: str) -> None:
        counters = self._fluxon_hostless_observation_counters
        anchor_locks_acquired = int(
            getattr(self, "_fluxon_hostless_anchor_locks_acquired", 0)
        )
        admission_acquired = int(
            getattr(self, "_fluxon_hostless_admission_acquired", 0)
        )
        if (
            not counters
            and not self._fluxon_hostless_request_observations
            and anchor_locks_acquired == 0
            and admission_acquired == 0
        ):
            return
        summary = ",".join(
            f"{key}={value}" for key, value in sorted(counters.items())
        )
        live_decisions: dict[str, int] = defaultdict(int)
        for observation in self._fluxon_hostless_request_observations.values():
            live_decisions[str(observation.get("prefetch_decision", "unknown"))] += 1
        live_summary = ",".join(
            f"{key}={value}" for key, value in sorted(live_decisions.items())
        )
        logger.info(
            "Fluxon hostless request lifecycle Snapshot: caller=%s tp_rank=%d "
            "live=%d live_decisions=%s counters=%s",
            caller,
            self._fluxon_hostless_tp_rank(),
            len(self._fluxon_hostless_request_observations),
            live_summary,
            summary,
        )
        logger.info(
            "Fluxon hostless anchor lock Snapshot: caller=%s tp_rank=%d "
            "active=%d acquired=%d released=%d high_watermark=%d",
            caller,
            self._fluxon_hostless_tp_rank(),
            int(getattr(self, "_fluxon_hostless_anchor_locks_active", 0)),
            anchor_locks_acquired,
            int(getattr(self, "_fluxon_hostless_anchor_locks_released", 0)),
            int(
                getattr(
                    self,
                    "_fluxon_hostless_anchor_locks_high_watermark",
                    0,
                )
            ),
        )
        self._log_fluxon_hostless_admission_snapshot(caller)

    def _new_fluxon_hostless_eviction_observation(
        self,
        requested_tokens: int,
    ) -> dict[str, Any]:
        return {
            "evict_requested_tokens": requested_tokens,
            "evict_actual_tokens": 0,
            "evict_candidate_tokens": 0,
            "evict_already_backed_tokens": 0,
            "evict_after_writeback_tokens": 0,
            "evict_unbacked_drop_tokens": 0,
            "evict_new_writebacks": 0,
            "evict_pending_writebacks": 0,
            "evict_write_backup_ms": 0.0,
            "evict_write_wait_ms": 0.0,
            "evict_free_group_ms": 0.0,
            "evict_total_ms": 0.0,
        }

    def _active_fluxon_hostless_eviction_observation(
        self,
    ) -> dict[str, Any] | None:
        if not self._fluxon_hostless_eviction_observation_stack:
            return None
        return self._fluxon_hostless_eviction_observation_stack[-1]

    def _has_fluxon_hostless_ready_entry(
        self,
        req: Any | None,
        node: UnifiedTreeNode,
        component_type: ComponentType = BASE_COMPONENT_TYPE,
        last_device_node: UnifiedTreeNode | None = None,
    ) -> bool:
        if not self._is_fluxon_hostless_full_mode() or req is None:
            return False
        operation = self.fluxon_hostless_ready_prefetch.get(req.rid)
        if operation is None:
            return False
        if component_type == BASE_COMPONENT_TYPE:
            if last_device_node is None:
                last_device_node = getattr(req, "last_node", None)
                if last_device_node is None:
                    return False
            hash_values = self._node_hash_values_after_ancestor(
                node,
                last_device_node,
            )
            if not hash_values:
                return False
            return self._fluxon_hostless_ready_operation_covers_kv(
                operation,
                hash_values,
            )
        if component_type == ComponentType.MAMBA:
            mamba_key = self._node_mamba_storage_key(node)
            return (
                mamba_key is not None
                and operation.mamba_plan_ptr is not None
                and operation.mamba_anchor_node_id == node.id
                and operation.mamba_key == mamba_key
            )
        return False

    def _fluxon_hostless_ready_operation_covers_kv(
        self,
        operation: _FluxonHostlessPrefetchOperation,
        hash_values: list[str],
        exact: bool = False,
    ) -> bool:
        if not hash_values:
            return False
        if operation.kv_plan_ptr is None:
            return False
        plan_offset_pages = int(
            getattr(operation, "kv_plan_offset_pages", 0)
        )
        if plan_offset_pages < 0:
            return False
        ready_pages = min(
            len(operation.hash_value),
            operation.completed_tokens // self.page_size,
        )
        plan_end_pages = plan_offset_pages + len(hash_values)
        if exact and ready_pages - plan_offset_pages != len(hash_values):
            return False
        if plan_end_pages > ready_pages:
            return False
        return (
            operation.hash_value[plan_offset_pages:plan_end_pages]
            == hash_values
        )

    def _fluxon_hostless_ready_value_ptrs(
        self,
        backend: Any,
        plan_ptr: int,
        operation: _FluxonHostlessPrefetchOperation,
        expected_count: int,
    ) -> tuple[int, ...]:
        plan_offset_pages = int(
            getattr(operation, "kv_plan_offset_pages", 0)
        )
        plan_end_pages = plan_offset_pages + int(expected_count)
        ready_pages = min(
            len(operation.hash_value),
            operation.completed_tokens // self.page_size,
        )
        if (
            plan_offset_pages < 0
            or expected_count <= 0
            or plan_end_pages > ready_pages
        ):
            raise RuntimeError(
                "Fluxon ready plan pointer slice is out of bounds: "
                f"offset={plan_offset_pages} count={expected_count} "
                f"ready={ready_pages}"
            )
        value_ptrs = backend.view_value_ptrs(plan_ptr, plan_end_pages)
        if len(value_ptrs) != plan_end_pages:
            raise RuntimeError(
                "Fluxon ready plan pointer view returned the wrong length: "
                f"expected={plan_end_pages} actual={len(value_ptrs)}"
            )
        return tuple(value_ptrs[plan_offset_pages:plan_end_pages])

    def _fluxon_hostless_longest_ready_restore_node(
        self,
        operation: _FluxonHostlessPrefetchOperation,
        best_match_node: UnifiedTreeNode,
        last_device_node: UnifiedTreeNode,
        failure_shape: dict[str, Any] | None = None,
    ) -> UnifiedTreeNode | None:
        """Return the deepest whole radix-node prefix covered by a ready plan.

        Fluxon ``get_start(prefix_best_effort=True)`` may prepare only a
        leading subset of an evicted path. Each radix node is submitted as an
        atomic group, so recovery must stop between nodes rather than split a
        node merely to consume every transferable page.
        """
        def _record_failure(reason: str, **fields: Any) -> None:
            if failure_shape is None:
                return
            failure_shape.clear()
            failure_shape.update(reason=reason, **fields)

        operation.kv_plan_offset_pages = 0

        if operation.kv_plan_ptr is None or not operation.hash_value:
            _record_failure(
                "invalid_plan",
                ready_pages=0,
                path_nodes=0,
                path_pages=0,
            )
            return None

        reverse_path: list[UnifiedTreeNode] = []
        cursor = best_match_node
        while cursor is not last_device_node:
            if cursor is self.root_node or cursor.parent is None:
                _record_failure(
                    "device_anchor_not_ancestor",
                    ready_pages=min(
                        len(operation.hash_value),
                        operation.completed_tokens // self.page_size,
                    ),
                    path_nodes=len(reverse_path),
                    path_pages=sum(len(node.key) for node in reverse_path)
                    // self.page_size,
                )
                return None
            reverse_path.append(cursor)
            cursor = cursor.parent

        completed_pages = operation.completed_tokens // self.page_size
        ready_pages = min(len(operation.hash_value), completed_pages)
        path_nodes = len(reverse_path)
        path_pages = sum(len(node.key) for node in reverse_path) // self.page_size
        if ready_pages <= 0:
            _record_failure(
                "zero_ready_pages",
                ready_pages=ready_pages,
                path_nodes=path_nodes,
                path_pages=path_pages,
            )
            return None
        if not reverse_path:
            _record_failure(
                "no_restore_path",
                ready_pages=ready_pages,
                path_nodes=0,
                path_pages=0,
            )
            return None

        first_node = reverse_path[-1]
        first_node_hashes = self._node_hash_values(first_node)
        if not first_node_hashes:
            _record_failure(
                "empty_node_hashes",
                ready_pages=ready_pages,
                path_nodes=path_nodes,
                path_pages=path_pages,
                failure_node_index=0,
                failure_node_pages=len(first_node.key) // self.page_size,
                consumed_pages=0,
                matched_pages=0,
                remaining_ready_pages=ready_pages,
                plan_offset_pages=0,
                alignment_candidates=0,
            )
            return None

        alignment_offsets = [
            index
            for index, operation_hash in enumerate(
                operation.hash_value[:ready_pages]
            )
            if operation_hash == first_node_hashes[0]
        ]
        if len(alignment_offsets) != 1:
            matched_pages = 0
            for operation_hash, node_hash in zip(
                operation.hash_value[:ready_pages], first_node_hashes
            ):
                if operation_hash != node_hash:
                    break
                matched_pages += 1
            _record_failure(
                (
                    "node_hash_mismatch"
                    if not alignment_offsets
                    else "ambiguous_plan_suffix_alignment"
                ),
                ready_pages=ready_pages,
                path_nodes=path_nodes,
                path_pages=path_pages,
                failure_node_index=0,
                failure_node_pages=len(first_node_hashes),
                consumed_pages=0,
                matched_pages=matched_pages,
                remaining_ready_pages=ready_pages,
                plan_offset_pages=0,
                alignment_candidates=len(alignment_offsets),
            )
            return None

        plan_offset_pages = alignment_offsets[0]
        operation.kv_plan_offset_pages = plan_offset_pages
        consumed_pages = 0
        ready_node: UnifiedTreeNode | None = None
        for node_index, node in enumerate(reversed(reverse_path)):
            node_hashes = self._node_hash_values(node)
            if not node_hashes:
                _record_failure(
                    "empty_node_hashes",
                    ready_pages=ready_pages,
                    path_nodes=path_nodes,
                    path_pages=path_pages,
                    failure_node_index=node_index,
                    failure_node_pages=len(node.key) // self.page_size,
                    consumed_pages=consumed_pages,
                    matched_pages=0,
                    remaining_ready_pages=(
                        ready_pages - plan_offset_pages - consumed_pages
                    ),
                    plan_offset_pages=plan_offset_pages,
                    alignment_candidates=1,
                )
                break
            next_pages = consumed_pages + len(node_hashes)
            if plan_offset_pages + next_pages > ready_pages:
                _record_failure(
                    "node_exceeds_ready_prefix",
                    ready_pages=ready_pages,
                    path_nodes=path_nodes,
                    path_pages=path_pages,
                    failure_node_index=node_index,
                    failure_node_pages=len(node_hashes),
                    consumed_pages=consumed_pages,
                    matched_pages=0,
                    remaining_ready_pages=(
                        ready_pages - plan_offset_pages - consumed_pages
                    ),
                    plan_offset_pages=plan_offset_pages,
                    alignment_candidates=1,
                )
                break
            operation_hashes = operation.hash_value[
                plan_offset_pages + consumed_pages : plan_offset_pages + next_pages
            ]
            if operation_hashes != node_hashes:
                matched_pages = 0
                for operation_hash, node_hash in zip(operation_hashes, node_hashes):
                    if operation_hash != node_hash:
                        break
                    matched_pages += 1
                _record_failure(
                    "node_hash_mismatch",
                    ready_pages=ready_pages,
                    path_nodes=path_nodes,
                    path_pages=path_pages,
                    failure_node_index=node_index,
                    failure_node_pages=len(node_hashes),
                    consumed_pages=consumed_pages,
                    matched_pages=matched_pages,
                    remaining_ready_pages=(
                        ready_pages - plan_offset_pages - consumed_pages
                    ),
                    plan_offset_pages=plan_offset_pages,
                    alignment_candidates=1,
                )
                break
            consumed_pages = next_pages
            ready_node = node
        if ready_node is None and failure_shape is not None and not failure_shape:
            _record_failure(
                "no_whole_node_unknown",
                ready_pages=ready_pages,
                path_nodes=path_nodes,
                path_pages=path_pages,
                consumed_pages=consumed_pages,
                plan_offset_pages=plan_offset_pages,
                alignment_candidates=1,
            )
        return ready_node

    def _full_kv_pool(self):
        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        return getattr(kvcache, "full_kv_pool", kvcache)

    def _full_kv_layers(self):
        full_kv_pool = self._full_kv_pool()
        get_key_buffer = getattr(full_kv_pool, "_get_key_buffer", None) or getattr(
            full_kv_pool, "get_key_buffer"
        )
        get_value_buffer = getattr(full_kv_pool, "_get_value_buffer", None) or getattr(
            full_kv_pool, "get_value_buffer"
        )
        start_layer = int(getattr(full_kv_pool, "start_layer", 0))
        layer_num = int(getattr(full_kv_pool, "layer_num", 0))
        return (
            tuple(get_key_buffer(start_layer + layer_id) for layer_id in range(layer_num)),
            tuple(
                get_value_buffer(start_layer + layer_id) for layer_id in range(layer_num)
            ),
        )

    def _mamba_pool(self):
        return getattr(self.req_to_token_pool, "mamba_pool", None)

    @staticmethod
    def _is_component_storage_backed(cd: ComponentData) -> bool:
        return bool(cd.metadata.get("storage_backed", False))

    @staticmethod
    def _is_component_storage_remote_backed(cd: ComponentData) -> bool:
        return bool(cd.metadata.get("storage_backed", False))

    def _node_prefix_len(self, node: Optional[UnifiedTreeNode]) -> int:
        total = 0
        while node is not None and node is not self.root_node:
            if node.key is not None:
                total += len(node.key)
            node = node.parent
        return total

    def _node_mamba_storage_key(self, node: UnifiedTreeNode) -> Optional[str]:
        hash_values = self._node_hash_values(node)
        if not hash_values:
            return None
        return hash_values[-1]

    def _node_hash_values_after_ancestor(
        self,
        node: UnifiedTreeNode,
        ancestor: UnifiedTreeNode,
    ) -> list[str]:
        hash_values: list[str] = []
        cursor = node
        while cursor is not ancestor and cursor is not self.root_node:
            hash_values[:0] = self._node_hash_values(cursor)
            cursor = cursor.parent
        if cursor is not ancestor:
            return []
        return hash_values

    def _normalize_mamba_indices(self, indices: torch.Tensor) -> torch.Tensor:
        mamba_pool = self._mamba_pool()
        if mamba_pool is None:
            raise RuntimeError("Hostless Fluxon Mamba requires a MambaPool")
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        return indices.reshape(-1).to(device=mamba_pool.device, dtype=torch.int64)

    def _alloc_hostless_mamba_slot(self) -> Optional[torch.Tensor]:
        mamba_pool = self._mamba_pool()
        if mamba_pool is None:
            return None
        slot = mamba_pool.alloc(1)
        if slot is not None:
            return slot
        self.evict(EvictParams(num_tokens=0, mamba_num=1))
        return mamba_pool.alloc(1)

    def _mamba_state_tensors(self) -> list[torch.Tensor]:
        mamba_pool = self._mamba_pool()
        if mamba_pool is None:
            raise RuntimeError("Hostless Fluxon Mamba requires a MambaPool")
        return [mamba_pool.mamba_cache.temporal, *mamba_pool.mamba_cache.conv]

    def _hostless_mamba_slot_index(self, mamba_indices: torch.Tensor) -> int:
        normalized_indices = self._normalize_mamba_indices(mamba_indices)
        if int(normalized_indices.numel()) != 1:
            raise RuntimeError(
                "Fluxon hostless Mamba owner path requires exactly one slot per node, "
                f"got {int(normalized_indices.numel())}"
            )
        return int(normalized_indices[0].item())

    def _fluxon_hostless_mamba_plan(self) -> dict[str, Any]:
        cached = self._fluxon_hostless_mamba_plan_cache
        state_tensors = self._mamba_state_tensors()
        layer_num = int(state_tensors[0].shape[0])
        cache_key = (
            tuple((tuple(t.shape), str(t.dtype), int(t.data_ptr())) for t in state_tensors),
            layer_num,
        )
        if cached is not None and cached["cache_key"] == cache_key:
            return cached

        state_item_bytes: list[int] = []
        state_layer_ptrs: list[int] = []
        for state_tensor in state_tensors:
            if int(state_tensor.shape[0]) != layer_num:
                raise RuntimeError(
                    "Hostless Fluxon Mamba requires identical layer counts across state tensors"
                )
            if not state_tensor[0].is_contiguous():
                raise RuntimeError(
                    "Hostless Fluxon Mamba owner path requires contiguous per-layer state tensors"
                )
            per_slot = state_tensor[0].narrow(0, 0, 1)
            state_item_bytes.append(int(per_slot.nbytes))
            for layer_idx in range(layer_num):
                if not state_tensor[layer_idx].is_contiguous():
                    raise RuntimeError(
                        "Hostless Fluxon Mamba owner path requires contiguous layer targets"
                    )
                state_layer_ptrs.append(int(state_tensor[layer_idx].data_ptr()))

        plan = {
            "cache_key": cache_key,
            "layer_num": layer_num,
            "state_count": len(state_tensors),
            "state_item_bytes": torch.tensor(
                state_item_bytes,
                dtype=torch.int64,
                device=self.device,
            ).contiguous(),
            "state_layer_ptrs": torch.tensor(
                state_layer_ptrs,
                dtype=torch.int64,
                device=self.device,
            ).contiguous(),
            "total_bytes": sum(layer_num * item_bytes for item_bytes in state_item_bytes),
        }
        self._fluxon_hostless_mamba_plan_cache = plan
        return plan

    def _hostless_mamba_needs_restore(self, node: UnifiedTreeNode) -> bool:
        if not self._is_fluxon_hostless_mamba_mode() or node is self.root_node:
            return False
        cd = node.component_data[ComponentType.MAMBA]
        return cd.value is None and self._is_component_storage_backed(cd)

    def _prepare_hostless_mamba_req_cow(
        self,
        req,
        source_indices: torch.Tensor,
        dst_index: Optional[torch.Tensor] = None,
    ) -> None:
        if req is None:
            return
        if req.mamba_pool_idx is None:
            if dst_index is None:
                dst_index = self._alloc_hostless_mamba_slot()
            assert (
                dst_index is not None
            ), "Cannot alloc request Mamba cache for Fluxon restore"
            req.mamba_pool_idx = dst_index[0]
        req.mamba_cow_src_index = self._normalize_mamba_indices(source_indices).clone()
        req.mamba_needs_clear = False

    def _commit_restored_hostless_mamba_state(
        self,
        node: UnifiedTreeNode,
        mamba_indices: torch.Tensor,
        req=None,
        req_dst_index: Optional[torch.Tensor] = None,
    ) -> None:
        ct = ComponentType.MAMBA
        cd = node.component_data[ct]
        cd.value = mamba_indices.clone()
        lru = self.lru_lists[ct]
        if lru.in_list(node):
            lru.reset_node_mru(node)
        else:
            lru.insert_mru(node)
        self.component_evictable_size_[ct] += len(mamba_indices)
        self._prepare_hostless_mamba_req_cow(
            req,
            mamba_indices,
            dst_index=req_dst_index,
        )
        self._update_evictable_leaf_sets(node)

    def _rollback_restored_hostless_mamba_state(
        self,
        node: UnifiedTreeNode,
        mamba_indices: torch.Tensor,
    ) -> None:
        ct = ComponentType.MAMBA
        cd = node.component_data[ct]
        if cd.value is not None:
            cd.value = None
            self.component_evictable_size_[ct] -= len(mamba_indices)
        lru = self.lru_lists[ct]
        if lru.in_list(node):
            lru.remove_node(node)
        mamba_pool = self._mamba_pool()
        if mamba_pool is not None:
            mamba_pool.free(mamba_indices)
        self._update_evictable_leaf_sets(node)

    def _restore_hostless_mamba_state(
        self,
        node: UnifiedTreeNode,
        backend,
        state_plan_ptr: int | None,
        h2d_submit_state: _FluxonRawH2DSubmitState,
        req=None,
    ) -> torch.Tensor:
        state_key = self._node_mamba_storage_key(node)
        if state_key is None:
            raise RuntimeError(
                f"Hostless Fluxon Mamba restore requires node hash: node={node.id}"
            )
        req_dst_index = None
        if req is not None and req.mamba_pool_idx is None:
            req_dst_index = self._alloc_hostless_mamba_slot()
            if req_dst_index is None:
                raise RuntimeError("Cannot alloc request Mamba cache for Fluxon restore")
        mamba_indices = self._alloc_hostless_mamba_slot()
        if mamba_indices is None:
            if req_dst_index is not None:
                mamba_pool = self._mamba_pool()
                if mamba_pool is not None:
                    mamba_pool.free(req_dst_index)
            raise RuntimeError("Cannot alloc Mamba cache for Fluxon restore")
        try:
            if node.id in self.ongoing_fluxon_hostless_backup:
                self._finish_fluxon_hostless_write_batch(
                    node.id,
                    block=True,
                    caller="_restore_hostless_mamba_state",
            )
            mamba_plan = self._fluxon_hostless_mamba_plan()
            mamba_slot_index = self._hostless_mamba_slot_index(mamba_indices)
            if state_plan_ptr is None:
                raise RuntimeError(
                    f"Hostless Fluxon Mamba restore requires a ready plan: node={node.id}"
                )
            try:
                restore_mamba_state_from_fluxon_values(
                    state_plan_ptr,
                    mamba_slot_index,
                    mamba_plan["state_layer_ptrs"],
                    mamba_plan["state_item_bytes"],
                    int(mamba_plan["layer_num"]),
                    self._cuda_device_index(),
                )
            except Exception:
                backend.release_views(state_plan_ptr)
                raise
            h2d_submit_state.add_finalizer(
                lambda plan_ptr=state_plan_ptr, release=backend.release_views: release(
                    plan_ptr
                )
            )
            h2d_submit_state.mark_pending()
            self._commit_restored_hostless_mamba_state(
                node,
                mamba_indices,
                req=req,
                req_dst_index=req_dst_index,
            )
            return mamba_indices
        except Exception:
            h2d_submit_state.synchronize()
            cd = node.component_data[ComponentType.MAMBA]
            if cd.value is not None and torch.equal(cd.value, mamba_indices):
                self._rollback_restored_hostless_mamba_state(node, mamba_indices)
            else:
                mamba_pool = self._mamba_pool()
                if mamba_pool is not None:
                    mamba_pool.free(mamba_indices)
            if req is not None and req.mamba_cow_src_index is not None:
                req.mamba_cow_src_index = None
            if req is not None and req_dst_index is not None:
                if (
                    req.mamba_pool_idx is not None
                    and torch.equal(req.mamba_pool_idx.unsqueeze(0), req_dst_index)
                ):
                    req.mamba_pool_idx = None
                mamba_pool = self._mamba_pool()
                if mamba_pool is not None:
                    mamba_pool.free(req_dst_index)
            raise

    def _fluxon_hostless_plan(self):
        cached = self._fluxon_hostless_plan_cache
        full_kv_pool = self._full_kv_pool()
        layer_num = int(getattr(full_kv_pool, "layer_num", 0))
        cache_key = (
            id(full_kv_pool),
            type(full_kv_pool),
            self.page_size,
            layer_num,
            str(getattr(full_kv_pool, "store_dtype", None)),
            int(getattr(full_kv_pool, "head_num", 0)),
            int(getattr(full_kv_pool, "head_dim", 0)),
            int(getattr(full_kv_pool, "v_head_dim", 0)),
            int(getattr(full_kv_pool, "kv_cache_dim", 0)),
        )
        if cached is not None and cached["cache_key"] == cache_key:
            return cached

        is_mla = isinstance(full_kv_pool, MLATokenToKVPool)
        if is_mla:
            kv_cache_dim = int(getattr(full_kv_pool, "kv_cache_dim"))
            kv_dtype = getattr(full_kv_pool, "store_dtype")
            item_size = kv_cache_dim * kv_dtype.itemsize
            use_jit = current_platform.is_cuda_alike() and can_use_hicache_jit_kernel(
                element_size=item_size
            )
            fragment_num = layer_num
            fragment_numel = self.page_size * kv_cache_dim * kv_dtype.itemsize
            total_bytes = fragment_num * fragment_numel
            plan = {
                "cache_key": cache_key,
                "is_mla": True,
                "layer_num": layer_num,
                "fragment_num": fragment_num,
                "fragment_numel": fragment_numel,
                "fragment_numels": tuple([fragment_numel] * layer_num),
                "total_bytes": total_bytes,
                "page_value_path": True,
                "page_bytes": fragment_numel,
                "layer_ptrs": full_kv_pool.data_ptrs.to(
                    device=self.device, dtype=torch.int64
                ).contiguous(),
                "layer_ptr_values": tuple(
                    int(layer.data_ptr()) for layer in full_kv_pool.kv_buffer
                ),
                "use_fast_path": True,
                "use_jit": use_jit,
                "stage_shape": (layer_num, self.page_size, 1, kv_cache_dim),
                "stage_dtype": kv_dtype,
                "fallback_kernel": getattr(
                    memory_pool_host_mod, "transfer_kv_all_layer_mla", None
                ),
                "item_size": item_size,
            }
        else:
            head_num = int(getattr(full_kv_pool, "head_num"))
            head_dim = int(getattr(full_kv_pool, "head_dim"))
            v_head_dim = int(getattr(full_kv_pool, "v_head_dim"))
            k_dtype = getattr(full_kv_pool, "store_dtype")
            v_dtype = getattr(full_kv_pool, "store_dtype")
            fast_put_get = head_dim == v_head_dim
            fragment_num = layer_num * 2
            fragment_numels = (
                [self.page_size * head_num * head_dim * k_dtype.itemsize] * layer_num
                + [self.page_size * head_num * v_head_dim * v_dtype.itemsize] * layer_num
            )
            k_page_bytes = self.page_size * head_num * head_dim * k_dtype.itemsize
            v_page_bytes = self.page_size * head_num * v_head_dim * v_dtype.itemsize
            total_bytes = sum(fragment_numels)
            plan = {
                "cache_key": cache_key,
                "is_mla": False,
                "layer_num": layer_num,
                "fragment_num": fragment_num,
                "fragment_numels": tuple(fragment_numels),
                "total_bytes": total_bytes,
                "page_value_path": True,
                "k_page_bytes": k_page_bytes,
                "v_page_bytes": v_page_bytes,
                "k_layer_ptrs": full_kv_pool.k_data_ptrs.to(
                    device=self.device, dtype=torch.int64
                ).contiguous(),
                "v_layer_ptrs": full_kv_pool.v_data_ptrs.to(
                    device=self.device, dtype=torch.int64
                ).contiguous(),
                "k_layer_ptr_values": tuple(
                    int(layer.data_ptr()) for layer in full_kv_pool.k_buffer
                ),
                "v_layer_ptr_values": tuple(
                    int(layer.data_ptr()) for layer in full_kv_pool.v_buffer
                ),
                "use_fast_path": False,
            }
            if fast_put_get:
                item_size = head_num * head_dim * k_dtype.itemsize
                use_jit = current_platform.is_cuda_alike() and can_use_hicache_jit_kernel(
                    element_size=item_size
                )
                plan.update(
                    {
                        "use_fast_path": True,
                        "use_jit": use_jit,
                        "k_stage_shape": (layer_num, self.page_size, head_num, head_dim),
                        "v_stage_shape": (layer_num, self.page_size, head_num, v_head_dim),
                        "k_stage_dtype": k_dtype,
                        "v_stage_dtype": v_dtype,
                        "fallback_kernel": getattr(
                            memory_pool_host_mod, "transfer_kv_all_layer", None
                        ),
                        "item_size": item_size,
                    }
                )

        if getattr(
            self, "_fluxon_hostless_eviction_write_stream_enabled", False
        ) and _is_cuda:
            plan_ready_event = torch.cuda.Event()
            plan_ready_event.record(torch.cuda.current_stream(device=self.device))
            plan["ready_event"] = plan_ready_event
        self._fluxon_hostless_plan_cache = plan
        return plan

    def _cuda_device_index(self) -> int:
        resolved_device_id = getattr(
            self, "_fluxon_hostless_cuda_device_id", None
        )
        if resolved_device_id is not None:
            return int(resolved_device_id)
        if isinstance(self.device, str):
            device_index_attr = torch.device(self.device).index
        elif isinstance(self.device, int):
            device_index_attr = self.device
        else:
            device_index_attr = getattr(self.device, "index", None)
            if callable(device_index_attr):
                device_index_attr = device_index_attr()
        return (
            torch.cuda.current_device()
            if device_index_attr is None
            else int(device_index_attr)
        )

    def _hostless_full_page_buffers(
        self,
        plan: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, ...]:
        if plan is None:
            plan = self._fluxon_hostless_plan()
        key_buffers, value_buffers = self._full_kv_layers()
        if plan["is_mla"]:
            return key_buffers
        return key_buffers + value_buffers

    def _alloc_hostless_full_stage(
        self,
        plan: dict[str, Any],
        num_slots: int,
    ) -> dict[str, torch.Tensor]:
        if num_slots <= 0:
            raise RuntimeError(
                f"Hostless Fluxon stage allocation requires num_slots > 0, got {num_slots}"
            )
        if plan["is_mla"]:
            shape = (
                plan["stage_shape"][0],
                num_slots,
                *plan["stage_shape"][2:],
            )
            return {
                "kv_stage": torch.empty(
                    shape,
                    dtype=plan["stage_dtype"],
                    device="cpu",
                    pin_memory=True,
                )
            }
        return {
            "k_stage": torch.empty(
                (
                    plan["k_stage_shape"][0],
                    num_slots,
                    *plan["k_stage_shape"][2:],
                ),
                dtype=plan["k_stage_dtype"],
                device="cpu",
                pin_memory=True,
            ),
            "v_stage": torch.empty(
                (
                    plan["v_stage_shape"][0],
                    num_slots,
                    *plan["v_stage_shape"][2:],
                ),
                dtype=plan["v_stage_dtype"],
                device="cpu",
                pin_memory=True,
            ),
        }

    def _fill_hostless_full_stage_from_device(
        self,
        slots: torch.Tensor,
        plan: dict[str, Any],
        stage: dict[str, torch.Tensor],
    ) -> None:
        full_kv_pool = self._full_kv_pool()
        num_slots = int(slots.numel())
        dst_indices = torch.arange(num_slots, dtype=torch.int64, device=self.device)

        if plan["is_mla"]:
            kv_stage = stage["kv_stage"]
            if plan["use_fast_path"]:
                stage_ptrs = torch.tensor(
                    [kv_stage[layer_id].data_ptr() for layer_id in range(plan["layer_num"])],
                    dtype=torch.uint64,
                    device=self.device,
                )
                if plan["use_jit"]:
                    jit_transfer_hicache_all_layer_mla(
                        ptr_dst=stage_ptrs,
                        indices_dst=dst_indices,
                        ptr_src=full_kv_pool.data_ptrs,
                        indices_src=slots,
                        cache_src_stride_bytes=plan["item_size"],
                        cache_dst_stride_bytes=plan["item_size"],
                        element_size=plan["item_size"],
                    )
                elif plan["fallback_kernel"] is not None:
                    plan["fallback_kernel"](
                        src_layers=full_kv_pool.data_ptrs,
                        dst_layers=stage_ptrs,
                        src_indices=slots,
                        dst_indices=dst_indices,
                        item_size=plan["item_size"],
                        num_layers=plan["layer_num"],
                    )
                else:
                    raise RuntimeError(
                        "Hostless Fluxon MLA fast-path requested but no kernel is available"
                    )
            else:
                src_layers = self._hostless_full_page_buffers(plan)
                for layer_id, src_layer in enumerate(src_layers):
                    kv_stage[layer_id].copy_(
                        src_layer.index_select(0, slots),
                        non_blocking=True,
                    )
            current_platform.synchronize()
            return

        key_buffers, value_buffers = self._full_kv_layers()
        k_stage = stage["k_stage"]
        v_stage = stage["v_stage"]
        if plan["use_fast_path"]:
            k_stage_ptrs = torch.tensor(
                [k_stage[layer_id].data_ptr() for layer_id in range(plan["layer_num"])],
                dtype=torch.uint64,
                device=self.device,
            )
            v_stage_ptrs = torch.tensor(
                [v_stage[layer_id].data_ptr() for layer_id in range(plan["layer_num"])],
                dtype=torch.uint64,
                device=self.device,
            )
            if plan["use_jit"]:
                jit_transfer_hicache_all_layer(
                    k_ptr_dst=k_stage_ptrs,
                    v_ptr_dst=v_stage_ptrs,
                    indices_dst=dst_indices,
                    k_ptr_src=full_kv_pool.k_data_ptrs,
                    v_ptr_src=full_kv_pool.v_data_ptrs,
                    indices_src=slots,
                    kv_cache_src_stride_bytes=plan["item_size"],
                    kv_cache_dst_stride_bytes=plan["item_size"],
                    element_size=plan["item_size"],
                )
            elif plan["fallback_kernel"] is not None:
                plan["fallback_kernel"](
                    src_k_layers=full_kv_pool.k_data_ptrs,
                    dst_k_layers=k_stage_ptrs,
                    src_v_layers=full_kv_pool.v_data_ptrs,
                    dst_v_layers=v_stage_ptrs,
                    src_indices=slots,
                    dst_indices=dst_indices,
                    item_size=plan["item_size"],
                    num_layers=plan["layer_num"],
                )
            else:
                raise RuntimeError(
                    "Hostless Fluxon MHA fast-path requested but no kernel is available"
                )
        else:
            for layer_id, src_layer in enumerate(key_buffers):
                k_stage[layer_id].copy_(
                    src_layer.index_select(0, slots),
                    non_blocking=True,
                )
            for layer_id, src_layer in enumerate(value_buffers):
                v_stage[layer_id].copy_(
                    src_layer.index_select(0, slots),
                    non_blocking=True,
                )
        current_platform.synchronize()

    def _slice_hostless_full_stage_page(
        self,
        plan: dict[str, Any],
        stage: dict[str, torch.Tensor],
        page_start: int,
    ) -> tuple[torch.Tensor, ...]:
        if plan["is_mla"]:
            kv_stage = stage["kv_stage"]
            return tuple(
                kv_stage[layer_id]
                .narrow(0, page_start, self.page_size)
                .view(torch.uint8)
                .reshape(-1)
                for layer_id in range(plan["layer_num"])
            )
        k_stage = stage["k_stage"]
        v_stage = stage["v_stage"]
        return tuple(
            k_stage[layer_id]
            .narrow(0, page_start, self.page_size)
            .view(torch.uint8)
            .reshape(-1)
            for layer_id in range(plan["layer_num"])
        ) + tuple(
            v_stage[layer_id]
            .narrow(0, page_start, self.page_size)
            .view(torch.uint8)
            .reshape(-1)
            for layer_id in range(plan["layer_num"])
        )

    def _stage_hostless_full_pages(
        self,
        slots: torch.Tensor,
        node: UnifiedTreeNode,
    ) -> list[_FluxonHostlessPageStage]:
        hash_values = self._node_hash_values(node)
        if not hash_values:
            return []
        self._hostless_page_starts(slots, expected_pages=len(hash_values))
        plan = self._fluxon_hostless_plan()
        stage = self._alloc_hostless_full_stage(plan, int(slots.numel()))
        self._fill_hostless_full_stage_from_device(slots, plan, stage)
        page_stages: list[_FluxonHostlessPageStage] = []
        for page_idx, page_hash in enumerate(hash_values):
            page_start = page_idx * self.page_size
            page_stages.append(
                _FluxonHostlessPageStage(
                    page_hash=page_hash,
                    page_index=page_idx,
                    fragment_tensors=self._slice_hostless_full_stage_page(
                        plan,
                        stage,
                        page_start,
                    ),
                )
            )
        return page_stages

    def _enqueue_raw_h2d_batch(
        self,
        dst_ptrs: array,
        src_ptrs: array,
        size_bytes: array,
    ) -> None:
        if len(dst_ptrs) == 0:
            return
        if len(dst_ptrs) != len(src_ptrs) or len(dst_ptrs) != len(size_bytes):
            raise RuntimeError(
                "Fluxon raw H2D batch descriptor length mismatch: "
                f"dst={len(dst_ptrs)} src={len(src_ptrs)} sizes={len(size_bytes)}"
            )
        dst_tensor = torch.frombuffer(dst_ptrs, dtype=torch.int64, count=len(dst_ptrs))
        src_tensor = torch.frombuffer(src_ptrs, dtype=torch.int64, count=len(src_ptrs))
        size_tensor = torch.frombuffer(size_bytes, dtype=torch.int64, count=len(size_bytes))
        transfer_raw_h2d_batch(
            dst_tensor,
            src_tensor,
            size_tensor,
            self._cuda_device_index(),
        )

    def _num_node_pages(self, node: UnifiedTreeNode) -> int:
        if node.hash_value is not None:
            return len(node.hash_value)
        if node.key is None or len(node.key) == 0:
            return 0
        return len(node.key) if self.page_size == 1 else len(node.key) // self.page_size

    def _node_page_offset(self, node: UnifiedTreeNode) -> int:
        offset = 0
        cursor = node.parent
        while cursor is not None and cursor is not self.root_node:
            offset += self._num_node_pages(cursor)
            cursor = cursor.parent
        return offset

    def _node_hash_values(self, node: UnifiedTreeNode) -> list[str]:
        if node.hash_value is None and node.key is not None and len(node.key) > 0:
            node.hash_value = compute_node_hash_values(node, self.page_size)
        return node.hash_value or []

    def _fluxon_hostless_extra_info(
        self, node: UnifiedTreeNode
    ) -> HiCacheStorageExtraInfo:
        page_count = len(self._node_hash_values(node))
        if page_count <= 0:
            raise RuntimeError(
                f"Fluxon Put atomic group requires a non-empty radix node: node={node.id}"
            )
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )
        return HiCacheStorageExtraInfo(
            prefix_keys=prefix_keys,
            atomic_group_lens=[page_count],
        )

    def _hostless_page_starts(
        self,
        slots: torch.Tensor,
        expected_pages: Optional[int] = None,
    ) -> list[int]:
        num_slots = int(slots.numel())
        if num_slots % self.page_size != 0:
            raise RuntimeError(
                f"Hostless Fluxon slots must be page-aligned: num_slots={num_slots} page_size={self.page_size}"
            )
        slot_pages = slots.reshape(-1, self.page_size)
        expected_offsets = torch.arange(
            self.page_size,
            dtype=slots.dtype,
            device=slots.device,
        )
        if not torch.equal(slot_pages, slot_pages[:, :1] + expected_offsets):
            raise RuntimeError("Hostless Fluxon requires page-contiguous slots")
        page_starts = [int(x) for x in slot_pages[:, 0].tolist()]
        if expected_pages is not None and len(page_starts) != expected_pages:
            raise RuntimeError(
                "Hostless Fluxon page count mismatch: "
                f"expected={expected_pages} got={len(page_starts)}"
            )
        return page_starts

    def _should_validate_hostless_page_indices(self) -> bool:
        validate_every_n = getattr(
            self, "_fluxon_hostless_page_index_validate_every_n", 0
        )
        if validate_every_n <= 0:
            return False
        call_count = (
            getattr(self, "_fluxon_hostless_page_index_call_count", 0) + 1
        )
        self._fluxon_hostless_page_index_call_count = call_count
        return call_count % validate_every_n == 0

    def _hostless_page_indices(
        self,
        slots: torch.Tensor,
        expected_pages: Optional[int] = None,
    ) -> torch.Tensor:
        num_slots = int(slots.numel())
        if num_slots % self.page_size != 0:
            raise RuntimeError(
                f"Hostless Fluxon slots must be page-aligned: num_slots={num_slots} page_size={self.page_size}"
            )
        slot_pages = slots.reshape(-1, self.page_size)
        page_starts = slot_pages[:, 0]
        if expected_pages is not None and int(page_starts.numel()) != expected_pages:
            raise RuntimeError(
                "Hostless Fluxon page count mismatch: "
                f"expected={expected_pages} got={int(page_starts.numel())}"
            )
        if self.page_size == 1:
            return page_starts.to(dtype=torch.int64).contiguous()

        # Page-contiguous/aligned slots are an allocator invariant.  The old
        # unconditional torch.equal/torch.any checks converted CUDA results to
        # Python bools and therefore synchronized the active model stream on
        # every Fluxon write-back.  Keep the checks available for explicit
        # debugging/sampling, but leave the production path GPU-asynchronous.
        if self._should_validate_hostless_page_indices():
            expected_offsets = torch.arange(
                self.page_size,
                dtype=slots.dtype,
                device=slots.device,
            )
            if not torch.equal(slot_pages, slot_pages[:, :1] + expected_offsets):
                raise RuntimeError("Hostless Fluxon requires page-contiguous slots")
            if torch.any(torch.remainder(page_starts, self.page_size) != 0):
                raise RuntimeError(
                    "Hostless Fluxon requires page starts aligned to page_size"
                )
        page_indices = torch.div(page_starts, self.page_size, rounding_mode="floor")
        return page_indices.to(dtype=torch.int64).contiguous()

    def _iter_hostless_full_pages(
        self,
        slots: torch.Tensor,
        node: UnifiedTreeNode,
    ):
        for page_stage in self._stage_hostless_full_pages(slots, node):
            yield (
                page_stage.page_hash,
                page_stage.page_index,
                list(page_stage.fragment_tensors),
            )

    def _hostless_mamba_storage_ready(self, node: UnifiedTreeNode) -> bool:
        if not self._is_fluxon_hostless_mamba_mode():
            return True
        cd = node.component_data[ComponentType.MAMBA]
        if cd.value is None and node.children:
            return True
        return self._is_component_storage_remote_backed(cd)

    def _fluxon_hostless_node_storage_ready(self, node: UnifiedTreeNode) -> bool:
        full_cd = node.component_data[BASE_COMPONENT_TYPE]
        if not self._is_component_storage_remote_backed(full_cd):
            return False
        return self._hostless_mamba_storage_ready(node)

    def _hostless_mamba_backup_key(
        self,
        node: UnifiedTreeNode,
    ) -> str | None:
        if not self._is_fluxon_hostless_mamba_mode():
            return None
        cd = node.component_data[ComponentType.MAMBA]
        if cd.value is None or self._is_component_storage_backed(cd):
            return None
        state_key = self._node_mamba_storage_key(node)
        if state_key is None:
            raise RuntimeError(
                f"Hostless Fluxon Mamba backup requires node hash: node={node.id}"
            )
        return state_key

    def _await_fluxon_batch_codes(
        self,
        future: Any,
        expected_codes: int,
        label: str,
    ) -> list[int]:
        wait_result = future.wait()
        if not hasattr(wait_result, "is_ok"):
            raise RuntimeError(f"Fluxon {label} wait returned invalid result object")
        if not wait_result.is_ok():
            raise RuntimeError(f"Fluxon {label} wait failed: {wait_result.unwrap_error()}")
        codes = wait_result.unwrap()
        if not isinstance(codes, list) or len(codes) != expected_codes:
            raise RuntimeError(
                f"Fluxon {label} returned mismatched ret-code list: expected={expected_codes} got={codes!r}"
            )
        bad_codes = [int(code) for code in codes if int(code) != 0]
        if bad_codes:
            raise RuntimeError(
                f"Fluxon {label} returned non-zero ret codes: {bad_codes}"
            )
        return [int(code) for code in codes]

    @staticmethod
    def _fluxon_future_is_waiting(future: Any) -> bool:
        is_waiting = getattr(future, "is_waiting", None)
        if is_waiting is None:
            return True
        try:
            return bool(is_waiting())
        except Exception:
            logger.exception("Fluxon future is_waiting() failed")
            return False

    def _schedule_fluxon_hostless_ack(
        self, batch: _FluxonHostlessWriteBatch
    ) -> None:
        if batch.kv_future is None and batch.mamba_future is None:
            batch.clear_keepalives()
            return
        ack_id = (batch.node.id, id(batch))
        self.ongoing_fluxon_hostless_acks[ack_id] = _FluxonHostlessFutureAck(batch)

    def _has_pending_fluxon_hostless_ack(self, node_id: int) -> bool:
        return any(
            ack.node.id == node_id for ack in self.ongoing_fluxon_hostless_acks.values()
        )

    def _fluxon_missing_backup_indices(
        self,
        backend: Any,
        keys: list[str],
        component_name: Optional[Any] = None,
    ) -> tuple[list[int], float]:
        if not _FLUXON_HOSTLESS_USE_EXISTENCE_FILTER:
            return list(range(len(keys))), 0.0
        filter_missing = getattr(backend, "batch_missing_indices", None)
        if filter_missing is None:
            return list(range(len(keys))), 0.0
        start = time.perf_counter()
        missing_indices = filter_missing(keys, component_name=component_name)
        filter_ms = (time.perf_counter() - start) * 1000.0
        if len(missing_indices) > len(keys):
            raise RuntimeError(
                "Fluxon batch_missing_indices returned too many indices: "
                f"keys={len(keys)} missing={len(missing_indices)}"
            )
        if any(index < 0 or index >= len(keys) for index in missing_indices):
            raise RuntimeError(
                "Fluxon batch_missing_indices returned out-of-range index: "
                f"keys={len(keys)} missing={missing_indices}"
            )
        if len(set(missing_indices)) != len(missing_indices):
            raise RuntimeError(
                f"Fluxon batch_missing_indices returned duplicate indices: {missing_indices}"
            )
        return missing_indices, filter_ms

    @staticmethod
    def _fluxon_put_conflict_kind(err: BaseException) -> str | None:
        exists_error_names = {"KeyAlreadyExistsError"}
        exists_error_markers = ("KeyAlreadyExistsError", "Key already exists")
        inflight_error_names = {"KeyBeingWrittenError"}
        inflight_error_markers = ("KeyBeingWrittenError", "currently being written")

        seen: set[int] = set()
        current: BaseException | None = err
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            current_name = current.__class__.__name__
            current_text = str(current)
            if current_name in exists_error_names or any(
                marker in current_text for marker in exists_error_markers
            ):
                return "exists"
            if current_name in inflight_error_names or any(
                marker in current_text for marker in inflight_error_markers
            ):
                return "inflight"

            next_err = current.__cause__
            if next_err is None and not getattr(current, "__suppress_context__", False):
                next_err = current.__context__
            current = next_err if isinstance(next_err, BaseException) else None

        return None

    @classmethod
    def _is_fluxon_retryable_put_conflict(cls, err: BaseException) -> bool:
        return cls._fluxon_put_conflict_kind(err) is not None

    def _fluxon_local_fast_put_start_with_conflict_reconcile(
        self,
        node_id: int,
        backend: Any,
        keys: list[str],
        value_len: int,
        component_name: Optional[Any] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
        max_conflict_retries: int = 4,
        allow_replica_degrade: bool = True,
    ) -> _FluxonLocalFastPutStartResult:
        if not keys:
            return _FluxonLocalFastPutStartResult([], None, 0.0, 0.0)

        pending_indices = list(range(len(keys)))
        total_filter_ms = 0.0
        total_local_fast_put_start_ms = 0.0
        conflict_retries = 0

        while pending_indices:
            pending_keys = [keys[index] for index in pending_indices]
            replica_admitted_count = 0
            if allow_replica_degrade:
                replica_admitted_count = (
                    backend.local_fast_put_start_replica_admitted_count(
                        pending_keys,
                        component_name=component_name,
                        extra_info=extra_info,
                    )
                )
            local_fast_put_start_begin = time.perf_counter()
            try:
                plan_ptr = backend.local_fast_put_start(
                    pending_keys,
                    value_len,
                    component_name=component_name,
                    extra_info=extra_info,
                )
                total_local_fast_put_start_ms += (
                    time.perf_counter() - local_fast_put_start_begin
                ) * 1000.0
                if conflict_retries > 0:
                    logger.info(
                        "Fluxon local_fast_put_start conflict resolved: node=%d component=%s original_keys=%d "
                        "remaining_keys=%d retries=%d",
                        node_id,
                        component_name,
                        len(keys),
                        len(pending_keys),
                        conflict_retries,
                    )
                return _FluxonLocalFastPutStartResult(
                    pending_indices,
                    int(plan_ptr),
                    total_filter_ms,
                    total_local_fast_put_start_ms,
                )
            except Exception as err:
                total_local_fast_put_start_ms += (
                    time.perf_counter() - local_fast_put_start_begin
                ) * 1000.0
                conflict_kind = self._fluxon_put_conflict_kind(err)
                if conflict_kind is None:
                    if not allow_replica_degrade or replica_admitted_count <= 0:
                        raise
                    local_only_begin = time.perf_counter()
                    try:
                        plan_ptr = backend.local_fast_put_start_local_only(
                            pending_keys,
                            value_len,
                            component_name=component_name,
                            extra_info=extra_info,
                        )
                    except Exception as local_only_err:
                        raise RuntimeError(
                            "Fluxon local_fast_put_start failed and local-only "
                            "validation also failed: "
                            f"node={node_id} component={component_name} "
                            f"original_keys={len(keys)} remaining_keys={len(pending_keys)} "
                            f"local_only_error={local_only_err}"
                        ) from err
                    total_local_fast_put_start_ms += (
                        time.perf_counter() - local_only_begin
                    ) * 1000.0
                    replica_error = str(err)
                    logger.warning(
                        "Fluxon local_fast_put_start replica degraded to local-only: "
                        "node=%d component=%s original_keys=%d remaining_keys=%d "
                        "replica_admitted=%d retries=%d error=%s",
                        node_id,
                        component_name,
                        len(keys),
                        len(pending_keys),
                        replica_admitted_count,
                        conflict_retries,
                        replica_error,
                    )
                    return _FluxonLocalFastPutStartResult(
                        pending_indices,
                        int(plan_ptr),
                        total_filter_ms,
                        total_local_fast_put_start_ms,
                        replica_degraded=True,
                        replica_error=replica_error,
                    )

                conflict_retries += 1
                if (
                    conflict_kind == "exists"
                    and not _FLUXON_HOSTLESS_USE_EXISTENCE_FILTER
                ):
                    logger.info(
                        "Fluxon local_fast_put_start existing-key conflict treated as already-backed "
                        "with existence filter disabled: node=%d component=%s original_keys=%d retries=%d",
                        node_id,
                        component_name,
                        len(keys),
                        conflict_retries,
                    )
                    return _FluxonLocalFastPutStartResult(
                        [], None, total_filter_ms, total_local_fast_put_start_ms
                    )
                if conflict_retries > max_conflict_retries:
                    raise RuntimeError(
                        "Fluxon local_fast_put_start conflict retry exhausted: "
                        f"node={node_id} component={component_name} original_keys={len(keys)} "
                        f"remaining_keys={len(pending_keys)}"
                    ) from err

                retry_missing, retry_filter_ms = self._fluxon_missing_backup_indices(
                    backend,
                    pending_keys,
                    component_name=component_name,
                )
                total_filter_ms += retry_filter_ms
                if not retry_missing:
                    logger.info(
                        "Fluxon local_fast_put_start conflict reconciled to already-backed keys: "
                        "node=%d component=%s original_keys=%d retries=%d",
                        node_id,
                        component_name,
                        len(keys),
                        conflict_retries,
                    )
                    return _FluxonLocalFastPutStartResult(
                        [], None, total_filter_ms, total_local_fast_put_start_ms
                    )

                if len(retry_missing) != len(pending_keys):
                    raise RuntimeError(
                        "Fluxon conflict reconciliation cannot split a Put atomic group: "
                        f"node={node_id} component={component_name} "
                        f"before={len(pending_keys)} missing={len(retry_missing)}"
                    )
                next_pending = [pending_indices[index] for index in retry_missing]
                if len(next_pending) != len(pending_indices):
                    logger.info(
                        "Fluxon local_fast_put_start conflict filtered existing keys after recheck: "
                        "node=%d component=%s before=%d after=%d retries=%d",
                        node_id,
                        component_name,
                        len(pending_indices),
                        len(next_pending),
                        conflict_retries,
                    )
                else:
                    sleep_s = min(0.001 * conflict_retries, 0.01)
                    logger.warning(
                        "Fluxon local_fast_put_start conflict persisted after recheck: node=%d component=%s "
                        "keys=%d retries=%d sleep_ms=%.3f error=%s",
                        node_id,
                        component_name,
                        len(pending_indices),
                        conflict_retries,
                        sleep_s * 1000.0,
                        err,
                    )
                    time.sleep(sleep_s)
                pending_indices = next_pending

        return _FluxonLocalFastPutStartResult(
            [], None, total_filter_ms, total_local_fast_put_start_ms
        )

    def _drain_fluxon_hostless_acks(self, block: bool = False) -> None:
        if not self.ongoing_fluxon_hostless_acks:
            return
        while True:
            progressed = False
            for ack_id, ack in list(self.ongoing_fluxon_hostless_acks.items()):
                if not block:
                    waiting = False
                    if ack.kv_future is not None:
                        waiting = waiting or self._fluxon_future_is_waiting(
                            ack.kv_future
                        )
                    if ack.mamba_future is not None:
                        waiting = waiting or self._fluxon_future_is_waiting(
                            ack.mamba_future
                        )
                    if waiting:
                        continue

                try:
                    node = ack.node
                    full_cd = node.component_data[BASE_COMPONENT_TYPE]
                    mamba_cd = (
                        node.component_data[ComponentType.MAMBA]
                        if ComponentType.MAMBA in self.components
                        else None
                    )
                    kv_ok = ack.kv_future is None
                    mamba_ok = ack.mamba_future is None
                    error_messages: list[str] = []
                    if ack.kv_future is not None:
                        try:
                            self._await_fluxon_batch_codes(
                                ack.kv_future,
                                ack.page_count,
                                f"hostless kv ack node={ack.node.id}",
                            )
                            kv_ok = True
                        except Exception as err:
                            error_messages.append(f"kv={err}")
                    if ack.mamba_future is not None:
                        try:
                            self._await_fluxon_batch_codes(
                                ack.mamba_future,
                                1,
                                f"hostless mamba ack node={ack.node.id}",
                            )
                            mamba_ok = True
                        except Exception as err:
                            error_messages.append(f"mamba={err}")

                    if ack.kv_future is not None:
                        full_cd.metadata["storage_pending"] = False
                        full_cd.metadata["storage_staged"] = False
                        full_cd.metadata["storage_local_ready"] = False
                        if kv_ok:
                            full_cd.metadata["storage_backed"] = True
                            self._record_store_event(
                                node, medium=StorageMedium.EXTERNAL
                            )
                        else:
                            full_cd.metadata["storage_replica_degraded"] = False
                            full_cd.metadata.pop(
                                "storage_replica_degrade_reason", None
                            )
                    if ack.mamba_future is not None and mamba_cd is not None:
                        mamba_cd.metadata["storage_pending"] = False
                        mamba_cd.metadata["storage_staged"] = False
                        mamba_cd.metadata["storage_local_ready"] = False
                        if mamba_ok:
                            mamba_cd.metadata["storage_backed"] = True
                        else:
                            mamba_cd.metadata["storage_replica_degraded"] = False
                            mamba_cd.metadata.pop(
                                "storage_replica_degrade_reason", None
                            )

                    if kv_ok and mamba_ok:
                        if (
                            self.enable_storage_metrics
                            and self.storage_metrics_collector is not None
                        ):
                            self.storage_metrics_collector.log_backuped_tokens(
                                ack.token_count
                            )
                        logger.info(
                            "HiCache write_backup remote ack complete: node=%d "
                            "mode=fluxon_hostless_full pages=%d mamba=%s bytes=%d "
                            "mamba_bytes=%d duration_ms=%.3f",
                            ack.node.id,
                            ack.page_count,
                            ack.has_mamba,
                            ack.total_bytes,
                            ack.mamba_bytes,
                            (time.perf_counter() - ack.start_time) * 1000.0,
                        )
                    else:
                        raise RuntimeError(", ".join(error_messages))
                except Exception as err:
                    logger.warning(
                        "HiCache write_backup remote ack failed: node=%d "
                        "mode=fluxon_hostless_full pages=%d mamba=%s "
                        "duration_ms=%.3f error=%s",
                        ack.node.id,
                        ack.page_count,
                        ack.has_mamba,
                        (time.perf_counter() - ack.start_time) * 1000.0,
                        err,
                    )
                finally:
                    self.ongoing_fluxon_hostless_acks.pop(ack_id, None)
                    ack.clear_keepalives()
                    progressed = True

            if not block or not self.ongoing_fluxon_hostless_acks:
                return
            if not progressed:
                time.sleep(0.001)

    def _submit_fluxon_hostless_write_batch(
        self,
        node: UnifiedTreeNode,
        backend: Any,
        write_back: bool = False,
    ) -> _FluxonHostlessWriteBatch:
        submit_start_time = time.perf_counter()
        device_value = node.component_data[BASE_COMPONENT_TYPE].value
        if device_value is None or len(device_value) == 0:
            raise RuntimeError(
                f"Hostless Fluxon write-back requires non-empty device value: node={node.id}"
            )
        plan = self._fluxon_hostless_plan()
        kv_keys = self._node_hash_values(node)
        if not kv_keys:
            raise RuntimeError(
                f"Hostless Fluxon write-back requires non-empty page keys: node={node.id}"
            )

        mamba_key = self._hostless_mamba_backup_key(node)
        mamba_future = None
        mamba_bytes = 0
        mamba_plan_ptr: int | None = None
        kv_future = None
        kv_plan_ptr: int | None = None
        local_ready_event = None
        page_stages: list[Any] = []
        total_bytes = 0
        page_index_prep_ms = 0.0
        kv_exist_filter_ms = 0.0
        kv_local_fast_put_start_ms = 0.0
        kv_write_ms = 0.0
        stream_sync_ms = 0.0
        kv_local_fast_put_commit_ms = 0.0
        mamba_local_fast_put_start_ms = 0.0
        mamba_write_ms = 0.0
        mamba_local_fast_put_commit_ms = 0.0
        mamba_storage_backed = False
        replica_degraded = False
        replica_degrade_reasons: list[str] = []
        dedicated_write_stream, source_ready_event = (
            self._fluxon_hostless_eviction_stream_guard(node, write_back)
        )
        write_stream_context = None
        try:
            if dedicated_write_stream is not None:
                write_stream_context = torch.cuda.stream(dedicated_write_stream)
                write_stream_context.__enter__()
                source_ready_event.wait(dedicated_write_stream)
                plan_ready_event = plan.get("ready_event")
                if plan_ready_event is not None:
                    plan_ready_event.wait(dedicated_write_stream)
            needs_local_ready_event = False
            if not bool(plan["page_value_path"]):
                raise RuntimeError(
                    "Fluxon hostless write-back requires page-value local_fast_put_start"
                )

            storage_extra_info = self._fluxon_hostless_extra_info(node)
            mamba_storage_extra_info = HiCacheStorageExtraInfo(
                prefix_keys=storage_extra_info.prefix_keys,
                extra_info=storage_extra_info.extra_info,
                atomic_group_lens=[1],
            )
            original_kv_pages = len(kv_keys)
            write_kv_indices, kv_exist_filter_ms = self._fluxon_missing_backup_indices(
                backend, kv_keys
            )
            write_kv_keys = [kv_keys[index] for index in write_kv_indices]
            full_cd = node.component_data[BASE_COMPONENT_TYPE]
            total_pages, total_first, total_last, total_sig = _fluxon_key_signature(
                kv_keys
            )
            put_pages, put_first, put_last, put_sig = _fluxon_key_signature(
                write_kv_keys
            )
            logger.info(
                "Fluxon hostless write keys: node=%d write_back=%s evicted=%s "
                "backuped=%s storage_staged=%s storage_backed=%s "
                "total_pages=%d total_first_key=%s total_last_key=%s total_key_sig=%s "
                "put_pages=%d put_first_key=%s put_last_key=%s put_key_sig=%s",
                node.id,
                write_back,
                node.evicted,
                node.backuped,
                bool(full_cd.metadata.get("storage_staged", False)),
                bool(full_cd.metadata.get("storage_backed", False)),
                total_pages,
                total_first,
                total_last,
                total_sig,
                put_pages,
                put_first,
                put_last,
                put_sig,
            )
            if write_kv_keys and len(write_kv_keys) != original_kv_pages:
                raise RuntimeError(
                    "Fluxon hostless existence filtering cannot split a Put atomic group: "
                    f"node={node.id} total={original_kv_pages} missing={len(write_kv_keys)}"
                )
            if len(write_kv_keys) != original_kv_pages:
                logger.info(
                    "Fluxon hostless write-back filtered existing kv pages before local_fast_put_start: "
                    "node=%d total=%d missing=%d",
                    node.id,
                    original_kv_pages,
                    len(write_kv_keys),
                )

            value_len = int(plan["total_bytes"])
            if write_kv_keys:
                kv_start_result = (
                    self._fluxon_local_fast_put_start_with_conflict_reconcile(
                        node.id,
                        backend,
                        write_kv_keys,
                        value_len,
                        extra_info=storage_extra_info,
                    )
                )
                retry_write_kv_indices = kv_start_result.indices
                kv_plan_ptr = kv_start_result.plan_ptr
                kv_exist_filter_ms += kv_start_result.filter_ms
                kv_local_fast_put_start_ms += kv_start_result.start_ms
                if kv_start_result.replica_degraded:
                    replica_degraded = True
                    replica_degrade_reasons.append(
                        f"kv:{kv_start_result.replica_error}"
                    )
                if len(retry_write_kv_indices) != len(write_kv_keys):
                    write_kv_indices = [
                        write_kv_indices[index]
                        for index in retry_write_kv_indices
                    ]
                    write_kv_keys = [
                        write_kv_keys[index]
                        for index in retry_write_kv_indices
                    ]
                if retry_write_kv_indices:
                    prep_start = time.perf_counter()
                    page_indices = self._hostless_page_indices(
                        device_value, expected_pages=original_kv_pages
                    )
                    if len(write_kv_indices) != original_kv_pages:
                        missing_tensor = torch.tensor(
                            write_kv_indices,
                            dtype=torch.int64,
                            device=page_indices.device,
                        )
                        page_indices = page_indices.index_select(0, missing_tensor)
                    page_index_prep_ms = (time.perf_counter() - prep_start) * 1000.0
                    total_bytes = len(write_kv_keys) * value_len
                    write_start = time.perf_counter()
                    if plan["is_mla"]:
                        write_mla_pages_to_fluxon_values(
                            kv_plan_ptr,
                            page_indices,
                            plan["layer_ptrs"],
                            int(plan["page_bytes"]),
                            self._cuda_device_index(),
                        )
                    else:
                        write_mha_pages_to_fluxon_values(
                            kv_plan_ptr,
                            page_indices,
                            plan["k_layer_ptrs"],
                            plan["v_layer_ptrs"],
                            int(plan["k_page_bytes"]),
                            int(plan["v_page_bytes"]),
                            self._cuda_device_index(),
                        )
                    kv_write_ms = (time.perf_counter() - write_start) * 1000.0
                    needs_local_ready_event = True
            if mamba_key is not None:
                mamba_value = node.component_data[ComponentType.MAMBA].value
                if mamba_value is None:
                    raise RuntimeError(
                        f"Hostless Fluxon Mamba backup requires live slot indices: node={node.id}"
                    )
                write_mamba_indices, mamba_exist_filter_ms = (
                    self._fluxon_missing_backup_indices(
                        backend, [mamba_key], component_name=PoolName.MAMBA
                    )
                )
                kv_exist_filter_ms += mamba_exist_filter_ms
                if write_mamba_indices:
                    mamba_plan = self._fluxon_hostless_mamba_plan()
                    mamba_slot_index = self._hostless_mamba_slot_index(mamba_value)
                    mamba_bytes = int(mamba_plan["total_bytes"])
                    mamba_start_result = (
                        self._fluxon_local_fast_put_start_with_conflict_reconcile(
                            node.id,
                            backend,
                            [mamba_key],
                            mamba_bytes,
                            component_name=PoolName.MAMBA,
                            extra_info=mamba_storage_extra_info,
                        )
                    )
                    retry_write_mamba_indices = mamba_start_result.indices
                    mamba_plan_ptr = mamba_start_result.plan_ptr
                    kv_exist_filter_ms += mamba_start_result.filter_ms
                    mamba_local_fast_put_start_ms += mamba_start_result.start_ms
                    if mamba_start_result.replica_degraded:
                        replica_degraded = True
                        replica_degrade_reasons.append(
                            f"mamba:{mamba_start_result.replica_error}"
                        )
                    if retry_write_mamba_indices:
                        mamba_write_start = time.perf_counter()
                        write_mamba_state_to_fluxon_values(
                            mamba_plan_ptr,
                            mamba_slot_index,
                            mamba_plan["state_layer_ptrs"],
                            mamba_plan["state_item_bytes"],
                            int(mamba_plan["layer_num"]),
                            self._cuda_device_index(),
                        )
                        mamba_write_ms = (
                            time.perf_counter() - mamba_write_start
                        ) * 1000.0
                        needs_local_ready_event = True
                    else:
                        mamba_storage_backed = True
                        logger.info(
                            "Fluxon hostless write-back reconciled mamba duplicate local_fast_put_start to existing state: node=%d",
                            node.id,
                        )
                else:
                    mamba_storage_backed = True
                    logger.info(
                        "Fluxon hostless write-back skipped existing mamba state before local_fast_put_start: node=%d",
                        node.id,
                    )
            if needs_local_ready_event:
                local_ready_event = torch.cuda.Event()
                local_ready_event.record(torch.cuda.current_stream(device=self.device))
        except Exception:
            if mamba_plan_ptr is not None and mamba_future is None:
                try:
                    backend.put_abort(mamba_plan_ptr)
                except Exception as cleanup_err:
                    logger.warning(
                        "Fluxon hostless mamba put_abort failed after submit error: node=%d error=%s",
                        node.id,
                        cleanup_err,
                    )
            if mamba_future is not None:
                try:
                    self._await_fluxon_batch_codes(
                        mamba_future,
                        1,
                        f"hostless mamba submit cleanup node={node.id}",
                    )
                except Exception as cleanup_err:
                    logger.warning(
                        "Fluxon hostless mamba cleanup wait failed after submit error: node=%d error=%s",
                        node.id,
                        cleanup_err,
                    )
            if plan["page_value_path"] and kv_plan_ptr is not None and kv_future is None:
                try:
                    backend.put_abort(kv_plan_ptr)
                except Exception as cleanup_err:
                    logger.warning(
                        "Fluxon hostless put_abort failed after submit error: node=%d error=%s",
                        node.id,
                        cleanup_err,
                    )
            if kv_future is not None:
                try:
                    self._await_fluxon_batch_codes(
                        kv_future,
                        len(kv_keys),
                        f"hostless kv submit cleanup node={node.id}",
                    )
                except Exception as cleanup_err:
                    logger.warning(
                        "Fluxon hostless cleanup wait failed after submit error: node=%d error=%s",
                        node.id,
                        cleanup_err,
                    )
            raise
        finally:
            if write_stream_context is not None:
                write_stream_context.__exit__(None, None, None)

        node.component_data[BASE_COMPONENT_TYPE].metadata["storage_staged"] = True
        node.component_data[BASE_COMPONENT_TYPE].metadata["storage_pending"] = True
        node.component_data[BASE_COMPONENT_TYPE].metadata["storage_local_ready"] = False
        node.component_data[BASE_COMPONENT_TYPE].metadata[
            "storage_replica_degraded"
        ] = replica_degraded
        if replica_degraded:
            node.component_data[BASE_COMPONENT_TYPE].metadata[
                "storage_replica_degrade_reason"
            ] = "; ".join(replica_degrade_reasons)
        else:
            node.component_data[BASE_COMPONENT_TYPE].metadata.pop(
                "storage_replica_degrade_reason", None
            )
        if mamba_plan_ptr is not None:
            node.component_data[ComponentType.MAMBA].metadata["storage_staged"] = True
            node.component_data[ComponentType.MAMBA].metadata["storage_pending"] = True
            node.component_data[ComponentType.MAMBA].metadata["storage_local_ready"] = False
            node.component_data[ComponentType.MAMBA].metadata[
                "storage_replica_degraded"
            ] = replica_degraded
            if replica_degraded:
                node.component_data[ComponentType.MAMBA].metadata[
                    "storage_replica_degrade_reason"
                ] = "; ".join(replica_degrade_reasons)
            else:
                node.component_data[ComponentType.MAMBA].metadata.pop(
                    "storage_replica_degrade_reason", None
                )
        batch = _FluxonHostlessWriteBatch(
            node=node,
            kv_future=kv_future,
            kv_plan_ptr=kv_plan_ptr,
            page_stages=page_stages,
            page_count=len(write_kv_keys),
            token_count=len(write_kv_keys) * self.page_size,
            total_bytes=total_bytes,
            mamba_future=mamba_future,
            mamba_plan_ptr=mamba_plan_ptr,
            local_ready_event=local_ready_event,
            local_ready_committed=not needs_local_ready_event,
            write_back=write_back,
            mamba_bytes=mamba_bytes,
            start_time=submit_start_time,
            page_index_prep_ms=page_index_prep_ms,
            kv_local_fast_put_start_ms=kv_local_fast_put_start_ms,
            kv_write_ms=kv_write_ms,
            stream_sync_ms=stream_sync_ms,
            kv_exist_filter_ms=kv_exist_filter_ms,
            kv_local_fast_put_commit_ms=kv_local_fast_put_commit_ms,
            mamba_local_fast_put_start_ms=mamba_local_fast_put_start_ms,
            mamba_write_ms=mamba_write_ms,
            mamba_local_fast_put_commit_ms=mamba_local_fast_put_commit_ms,
            mamba_storage_backed=mamba_storage_backed,
            replica_degraded=replica_degraded,
            replica_degrade_reasons=replica_degrade_reasons,
            dedicated_write_stream=dedicated_write_stream is not None,
        )
        self.ongoing_fluxon_hostless_backup[node.id] = batch
        return batch

    def _finish_fluxon_hostless_write_batch(
        self,
        node_id: int,
        block: bool,
        caller: str = "unknown",
    ) -> bool:
        batch = self.ongoing_fluxon_hostless_backup.get(node_id)
        if batch is None:
            return False
        advanced = self._advance_fluxon_hostless_write_batch(
            node_id, block=block, caller=caller
        )
        if not advanced:
            return False

        node = batch.node
        full_cd = node.component_data[BASE_COMPONENT_TYPE]
        mamba_cd = (
            node.component_data[ComponentType.MAMBA]
            if ComponentType.MAMBA in self.components
            else None
        )
        success = False
        ack_scheduled = False
        metadata_mark_ms = 0.0
        pending_cleanup_ms = 0.0
        local_submit_ms = 0.0
        kv_ack_pending = batch.kv_future is not None
        kv_remote_ready = batch.page_count == 0
        mamba_ack_pending = batch.mamba_future is not None
        mamba_remote_ready = batch.mamba_storage_backed
        try:
            logger.info(
                "HiCache write_backup finish_enter: node=%d caller=%s block=%s "
                "storage_pending=%s storage_staged=%s storage_local_ready=%s",
                node.id,
                caller,
                block,
                bool(full_cd.metadata.get("storage_pending", False)),
                bool(full_cd.metadata.get("storage_staged", False)),
                bool(full_cd.metadata.get("storage_local_ready", False)),
            )
            metadata_mark_start = time.perf_counter()
            success = True
            if kv_remote_ready:
                full_cd.metadata["storage_backed"] = True
                self._record_store_event(node, medium=StorageMedium.EXTERNAL)
            if mamba_cd is not None and mamba_remote_ready:
                mamba_cd.metadata["storage_backed"] = True
            self._schedule_fluxon_hostless_ack(batch)
            ack_scheduled = True
            metadata_mark_ms = (time.perf_counter() - metadata_mark_start) * 1000.0
            local_submit_ms = (
                batch.page_index_prep_ms
                + batch.kv_exist_filter_ms
                + batch.kv_local_fast_put_start_ms
                + batch.kv_write_ms
                + batch.stream_sync_ms
                + batch.kv_local_fast_put_commit_ms
                + batch.mamba_local_fast_put_start_ms
                + batch.mamba_write_ms
                + batch.mamba_local_fast_put_commit_ms
            )
        except Exception as err:
            logger.warning(
                "HiCache write_backup failed: node=%d mode=fluxon_hostless_full "
                "caller=%s block=%s pages=%d mamba=%s duration_ms=%.3f error=%s",
                node.id,
                caller,
                block,
                batch.page_count,
                batch.mamba_future is not None or batch.mamba_storage_backed,
                (time.perf_counter() - batch.start_time) * 1000.0,
                err,
            )
        finally:
            pending_cleanup_start = time.perf_counter()
            if not success:
                full_cd.metadata["storage_pending"] = False
                full_cd.metadata["storage_local_ready"] = False
                full_cd.metadata["storage_staged"] = False
                full_cd.metadata["storage_replica_degraded"] = False
                full_cd.metadata.pop("storage_replica_degrade_reason", None)
            elif kv_ack_pending:
                full_cd.metadata["storage_pending"] = True
                full_cd.metadata["storage_staged"] = True
            else:
                full_cd.metadata["storage_pending"] = False
                full_cd.metadata["storage_local_ready"] = False
                full_cd.metadata["storage_staged"] = False
            if mamba_cd is not None and (
                batch.mamba_plan_ptr is not None
                or batch.mamba_future is not None
                or batch.mamba_storage_backed
            ):
                if not success:
                    mamba_cd.metadata["storage_pending"] = False
                    mamba_cd.metadata["storage_local_ready"] = False
                    mamba_cd.metadata["storage_staged"] = False
                    mamba_cd.metadata["storage_replica_degraded"] = False
                    mamba_cd.metadata.pop("storage_replica_degrade_reason", None)
                elif mamba_ack_pending:
                    mamba_cd.metadata["storage_pending"] = True
                    mamba_cd.metadata["storage_staged"] = True
                else:
                    mamba_cd.metadata["storage_pending"] = False
                    mamba_cd.metadata["storage_local_ready"] = False
                    mamba_cd.metadata["storage_staged"] = False
            self.ongoing_fluxon_hostless_backup.pop(node_id, None)
            if not ack_scheduled:
                batch.clear_keepalives()
            pending_cleanup_ms = (
                time.perf_counter() - pending_cleanup_start
            ) * 1000.0
            if success:
                evict_success_ms = (time.perf_counter() - batch.start_time) * 1000.0
                logger.info(
                    "HiCache write_backup complete: node=%d tokens=%d mode=fluxon_hostless_full "
                    "caller=%s block=%s "
                    "pages=%d mamba=%s bytes=%d mamba_bytes=%d page_index_prep_ms=%.3f "
                    "kv_exist_filter_ms=%.3f kv_local_fast_put_start_ms=%.3f "
                    "kv_write_ms=%.3f stream_sync_ms=%.3f "
                    "kv_local_fast_put_commit_ms=%.3f mamba_local_fast_put_start_ms=%.3f "
                    "mamba_write_ms=%.3f mamba_local_fast_put_commit_ms=%.3f "
                    "replica_degraded=%s replica_degrade_reason=%s "
                    "metadata_mark_ms=%.3f pending_cleanup_ms=%.3f "
                    "dedicated_write_stream=%s local_submit_ms=%.3f "
                    "evict_success_ms=%.3f duration_ms=%.3f",
                    node.id,
                    batch.token_count,
                    caller,
                    block,
                    batch.page_count,
                    batch.mamba_plan_ptr is not None
                    or batch.mamba_future is not None
                    or batch.mamba_storage_backed,
                    batch.total_bytes,
                    batch.mamba_bytes,
                    batch.page_index_prep_ms,
                    batch.kv_exist_filter_ms,
                    batch.kv_local_fast_put_start_ms,
                    batch.kv_write_ms,
                    batch.stream_sync_ms,
                    batch.kv_local_fast_put_commit_ms,
                    batch.mamba_local_fast_put_start_ms,
                    batch.mamba_write_ms,
                    batch.mamba_local_fast_put_commit_ms,
                    batch.replica_degraded,
                    "; ".join(batch.replica_degrade_reasons),
                    metadata_mark_ms,
                    pending_cleanup_ms,
                    batch.dedicated_write_stream,
                    local_submit_ms,
                    evict_success_ms,
                    evict_success_ms,
                )
            if (
                not success
                and node is not self.root_node
                and node.evicted
                and not node.backuped
                and len(node.children) == 0
                and all(
                    cd.lock_ref == 0 and cd.host_lock_ref == 0
                    for cd in node.component_data
                )
            ):
                tracker = {ct: 0 for ct in self.tree_components}
                self.evictable_device_leaves.discard(node)
                self.evictable_host_leaves.discard(node)
                self._remove_leaf_from_parent(node)
                self._iteratively_delete_tombstone_leaf(node, tracker)
        return success

    def _advance_fluxon_hostless_write_batch(
        self,
        node_id: int,
        block: bool,
        caller: str = "unknown",
    ) -> bool:
        batch = self.ongoing_fluxon_hostless_backup.get(node_id)
        if batch is None:
            return False

        if not batch.local_ready_committed:
            if batch.local_ready_event is None:
                batch.local_ready_committed = True
            else:
                if block:
                    sync_start = time.perf_counter()
                    batch.local_ready_event.synchronize()
                    batch.stream_sync_ms += (time.perf_counter() - sync_start) * 1000.0
                elif not bool(batch.local_ready_event.query()):
                    return False

                backend = self._fluxon_backend()
                if backend is None:
                    raise RuntimeError(
                        "Fluxon hostless backup lost backend while advancing local-ready batch"
                    )
                node = batch.node
                full_cd = node.component_data[BASE_COMPONENT_TYPE]
                mamba_cd = (
                    node.component_data[ComponentType.MAMBA]
                    if ComponentType.MAMBA in self.components
                    else None
                )
                kv_commit_start = time.perf_counter()
                if batch.kv_plan_ptr is not None and batch.kv_future is None:
                    batch.kv_future = backend.local_fast_put_commit(batch.kv_plan_ptr)
                    batch.kv_local_fast_put_commit_ms += (
                        time.perf_counter() - kv_commit_start
                    ) * 1000.0
                if batch.mamba_plan_ptr is not None and batch.mamba_future is None:
                    mamba_commit_start = time.perf_counter()
                    batch.mamba_future = backend.local_fast_put_commit(batch.mamba_plan_ptr)
                    batch.mamba_local_fast_put_commit_ms += (
                        time.perf_counter() - mamba_commit_start
                    ) * 1000.0
                batch.local_ready_committed = True
                full_cd.metadata["storage_local_ready"] = True
                if mamba_cd is not None and batch.mamba_plan_ptr is not None:
                    mamba_cd.metadata["storage_local_ready"] = True
        return True

    def _wait_for_fluxon_hostless_backup(self, node: UnifiedTreeNode) -> bool:
        while node.id in self.ongoing_fluxon_hostless_backup:
            self._finish_fluxon_hostless_write_batch(
                node.id,
                block=True,
                caller="_wait_for_fluxon_hostless_backup",
            )
        while self._has_pending_fluxon_hostless_ack(node.id):
            self._drain_fluxon_hostless_acks(block=True)
        return self._fluxon_hostless_node_storage_ready(node)

    def _wait_for_pending_fluxon_write_through(
        self, node: UnifiedTreeNode
    ) -> bool:
        has_pending = (
            node.id in self.ongoing_fluxon_hostless_backup
            or self._has_pending_fluxon_hostless_ack(node.id)
        )
        if not has_pending:
            return False
        return self._wait_for_fluxon_hostless_backup(node)

    def _ensure_fluxon_hostless_write_through_recoverable(
        self, node: UnifiedTreeNode
    ) -> bool:
        if self._wait_for_pending_fluxon_write_through(node):
            return True
        if self.write_backup(node, write_back=True) <= 0:
            return False
        self.writing_check(write_back=True)
        return self._fluxon_hostless_node_storage_ready(node)

    def _flush_pending_fluxon_hostless_backups(self) -> None:
        while self.ongoing_fluxon_hostless_backup:
            for node_id in list(self.ongoing_fluxon_hostless_backup):
                self._finish_fluxon_hostless_write_batch(
                    node_id,
                    block=True,
                    caller="_flush_pending_fluxon_hostless_backups",
                )
        self._drain_fluxon_hostless_acks(block=True)

    def _pending_fluxon_hostless_page_views(
        self,
        node: UnifiedTreeNode,
    ) -> list[_FluxonHostlessPageStage] | None:
        batch = self.ongoing_fluxon_hostless_backup.get(node.id)
        if batch is None:
            return None
        return batch.page_stages

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result

        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return self._empty_match_result
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        (
            value,
            best_match_node,
            best_match_value_len,
            best_match_device_node,
            best_match_device_value_len,
            mamba_state_node,
            mamba_state_value_len,
        ) = self._match_prefix_helper(key)
        return self._match_post_processor(
            params,
            value,
            best_match_node,
            best_match_value_len,
            best_match_device_node,
            best_match_device_value_len,
            mamba_state_node,
            mamba_state_value_len,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        result = self._insert_helper(self.root_node, key, value, params)
        return result

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        # A radix eviction commonly frees several tree nodes. Token allocators
        # implement free() with a torch.cat into the free-page tensor, so doing
        # that once per node creates repeated GPU allocations and copies. Use
        # the allocator's existing free-group contract to concatenate all node
        # indices once per top-level eviction. Preserve an outer free group if
        # the caller already owns one.
        allocator = self.token_to_kv_pool_allocator
        started_free_group = (
            allocator is not None and allocator.is_not_in_free_group
        )
        if started_free_group:
            free_group_started_at = time.perf_counter()
            allocator.free_group_begin()
            observation = self._active_fluxon_hostless_eviction_observation()
            if observation is not None:
                observation["evict_free_group_ms"] += (
                    time.perf_counter() - free_group_started_at
                ) * 1000.0
        try:
            for component in self._components_tuple:
                component.drive_eviction(params=params, tracker=tracker)
        finally:
            if started_free_group:
                free_group_started_at = time.perf_counter()
                allocator.free_group_end()
                observation = self._active_fluxon_hostless_eviction_observation()
                if observation is not None:
                    observation["evict_free_group_ms"] += (
                        time.perf_counter() - free_group_started_at
                    ) * 1000.0

        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            self.writing_check(write_back=not self._is_fluxon_hostless_full_mode())

        self.update_eviction_metrics(sum(tracker.values()), start_time)
        evicted_total = sum(tracker.values())
        if evicted_total > 0:
            logger.info(
                "HiCache evict result: requested=%d evicted=%d mamba_evicted=%d swa_evicted=%d total_components_evicted=%d",
                params.num_tokens,
                tracker[BASE_COMPONENT_TYPE],
                tracker.get(ComponentType.MAMBA, 0),
                tracker.get(ComponentType.SWA, 0),
                evicted_total,
            )
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        result = self.session.try_inc_lock_ref(node)
        if result is not None:
            return result
        if self.disable:
            return IncLockRefResult()
        result = IncLockRefResult()
        for component in self._components_tuple:
            result = component.acquire_component_lock(node=node, result=result)

        self._update_evictable_leaf_sets(node)
        return result

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        result = self.session.try_dec_lock_ref(node, params)
        if result is not None:
            return result
        if self.disable:
            return DecLockRefResult()
        for component in self._components_tuple:
            component.release_component_lock(node=node, params=params)

        self._update_evictable_leaf_sets(node)
        # TODO: delta is not aggregated from components; no caller uses it yet.
        return DecLockRefResult()

    def inc_host_lock_ref(self, node: Any) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult()
        result = IncLockRefResult()
        for component in self._components_tuple:
            result = component.acquire_component_lock(
                node=node, result=result, lock_host=True
            )

        self._update_evictable_leaf_sets(node)
        return result

    def dec_host_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult()
        for component in self._components_tuple:
            component.release_component_lock(node=node, params=params, lock_host=True)

        self._update_evictable_leaf_sets(node)
        return DecLockRefResult()

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs) -> None:
        if self.session.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
            return

        kv_committed_len = req.pop_committed_kv_cache()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(req, is_finished=True)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        result = None
        insert_params = None

        if is_insert:
            insert_params = InsertParams(
                prev_prefix_len=req.cache_protected_len,
                priority=getattr(req, "priority", 0) or 0,
            )

            # components prepare insert data + return effective cache_len
            effective_cache_len = len(token_ids)
            for comp in self._components_tuple:
                cl = comp.prepare_for_caching_req(
                    req=req,
                    insert_params=insert_params,
                    token_ids_len=len(token_ids),
                    is_finished=True,
                )
                if cl is not None:
                    effective_cache_len = min(effective_cache_len, cl)

            # Truncate if needed
            if effective_cache_len < len(token_ids):
                free_start = max(effective_cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[free_start:])
                token_ids = token_ids[:effective_cache_len]
                kv_indices = kv_indices[:effective_cache_len]

            radix_key = RadixKey(
                token_ids, req.extra_key, is_bigram=self.is_eagle
            ).page_aligned(self.page_size)
            page_aligned_len = len(radix_key)
            values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

            insert_params.key = radix_key
            insert_params.value = values
            result = self.insert(insert_params)

            # Free unaligned tail
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])

        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req, is_finished=True, insert_result=result, insert_params=insert_params
            )

    def cache_unfinished_req(self, req: Req, chunked=False, **kwargs) -> None:
        if self.session.try_cache_unfinished_req(req, chunked=chunked, **kwargs):
            return

        token_ids = req.fill_ids

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            req.prefix_indices = kv_indices
            return

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # components prepare insert data + return effective cache_len
        insert_params = InsertParams(
            prev_prefix_len=req.cache_protected_len,
            chunked=chunked,
            priority=getattr(req, "priority", 0) or 0,
        )
        effective_cache_len = len(token_ids)
        for comp in self._components_tuple:
            cl = comp.prepare_for_caching_req(
                req=req,
                insert_params=insert_params,
                token_ids_len=len(token_ids),
                is_finished=False,
            )
            if cl is not None:
                effective_cache_len = min(effective_cache_len, cl)

        if effective_cache_len <= 0:
            req.prefix_indices = kv_indices_orig.to(dtype=torch.int64, copy=True)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(
                    req, is_finished=False, insert_params=insert_params
                )
            return

        kv_indices = kv_indices_orig[:effective_cache_len]

        radix_key = RadixKey(
            token_ids[:effective_cache_len],
            req.extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        page_aligned_len = len(radix_key)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

        insert_params.key = radix_key
        insert_params.value = values
        result = self.insert(insert_params)

        # Match prefix
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        new_prefix_len = result.prefix_len
        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {page_aligned_len=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        lock_result = self.inc_lock_ref(new_last_node)

        # Update req fields
        if len(new_indices) < len(kv_indices_orig):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices_orig[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.cache_protected_len = len(new_indices)
        req.last_node = new_last_node
        req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req,
                is_finished=False,
                insert_result=result,
                insert_params=insert_params,
            )

    # ---- Internal Helpers ----

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> tuple[
        list[torch.Tensor],
        UnifiedTreeNode,
        int,
        UnifiedTreeNode,
        int,
        UnifiedTreeNode,
        int,
    ]:
        # Non-HiCache mode has only device-resident matches, so the scheduler
        # device anchor follows the best match. In HiCache mode, host-backed
        # nodes can also match, so we separately track the best device-resident
        # match for scheduler prefix indices and locking.
        node = self.root_node
        child_key = key.child_key(self.page_size)
        value: list[torch.Tensor] = []
        best_match_node = node
        best_match_value_len = 0
        best_match_device_node = node
        best_match_device_value_len = 0
        mamba_state_node = node
        mamba_state_value_len = 0
        separate_device_match = self.cache_controller is not None
        hostless_mamba = self._is_fluxon_hostless_mamba_mode()
        if hostless_mamba:
            kv_validator = self.components[
                BASE_COMPONENT_TYPE
            ].create_match_validator()
            kv_device_validator = self.components[
                BASE_COMPONENT_TYPE
            ].create_match_validator(match_device_only=True)
            mamba_validator = self.components[
                ComponentType.MAMBA
            ].create_match_validator()
        elif separate_device_match:
            validators = tuple(
                comp.create_match_validator() for comp in self._components_tuple
            )
            device_validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in self._components_tuple
            )
        else:
            validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in self._components_tuple
            )

        def _all_valid(validators, node):
            return all([v(node) for v in validators])

        def _update_best_if_valid(node):
            nonlocal best_match_node, best_match_value_len
            nonlocal best_match_device_value_len, best_match_device_node
            nonlocal mamba_state_node, mamba_state_value_len
            if hostless_mamba:
                if kv_validator(node):
                    best_match_node = node
                    best_match_value_len = len(value)
                if mamba_validator(node):
                    mamba_state_node = node
                    mamba_state_value_len = len(value)
                if kv_device_validator(node):
                    best_match_device_value_len = len(value)
                    best_match_device_node = node
                return

            matched = _all_valid(validators, node)
            if matched:
                best_match_node = node
                # This is the original Mamba boundary semantics: host/storage
                # backed state is a valid state boundary, even if the scheduler
                # can only directly use the shallower device-resident prefix.
                best_match_value_len = len(value)

            if not separate_device_match:
                if matched:
                    best_match_device_value_len = len(value)
                    best_match_device_node = node
                return
            if _all_valid(device_validators, node):
                best_match_device_value_len = len(value)
                best_match_device_node = node

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]

            # HiCache: dead node (evicted + not backuped) — stop traversal
            if child.evicted and not child.backuped:
                break

            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                node = self._split_node(child.key, child, prefix_len)
                if not node.evicted:
                    value.append(node.component_data[BASE_COMPONENT_TYPE].value)
                _update_best_if_valid(node)
                break

            if not child.evicted:
                value.append(child.component_data[BASE_COMPONENT_TYPE].value)
            node = child
            _update_best_if_valid(node)
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

        return (
            value,
            best_match_node,
            best_match_value_len,
            best_match_device_node,
            best_match_device_value_len,
            mamba_state_node if hostless_mamba else best_match_node,
            mamba_state_value_len if hostless_mamba else best_match_value_len,
        )

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        best_match_node: UnifiedTreeNode,
        best_match_value_len: int,
        best_match_device_node: UnifiedTreeNode,
        best_match_device_value_len: int,
        mamba_state_node: UnifiedTreeNode,
        mamba_state_value_len: int,
    ) -> MatchResult:
        node_update = best_match_node
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue  # Full uses last_access_time, not LRU
            refresh_node = (
                mamba_state_node
                if (
                    self._is_fluxon_hostless_mamba_mode()
                    and comp.component_type == ComponentType.MAMBA
                )
                else node_update
            )
            comp.refresh_lru(LRURefreshPhase.MATCH_END, refresh_node, self.root_node)

        cur_time = get_and_increase_time_counter()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001
            node_update = node_update.parent

        # last_host_node is the lower-tier KV anchor used by prefetch/load-back.
        # In hostless Mamba mode, Mamba state availability is tracked separately
        # by mamba_state_node and must not suppress the KV boundary.
        last_host_node = (
            best_match_node
            if self.cache_controller is not None
            else best_match_device_node
        )

        if best_match_device_value_len > 0:
            device_indices = torch.cat(value[:best_match_device_value_len])
        else:
            device_indices = self._empty_match_result.device_indices
        result = MatchResult(
            device_indices=device_indices,
            last_device_node=best_match_device_node,
            last_host_node=last_host_node,
            best_match_node=best_match_node,
            host_hit_length=0,
        )

        for component in self._components_tuple:
            component_best_value_len = (
                mamba_state_value_len
                if (
                    self._is_fluxon_hostless_mamba_mode()
                    and component.component_type == ComponentType.MAMBA
                )
                else best_match_value_len
            )
            result = component.finalize_match_result(
                result=result,
                params=params,
                value_chunks=value,
                best_value_len=component_best_value_len,
            )
        return result

    def _split_node(
        self, key: RadixKey, child: UnifiedTreeNode, split_len: int
    ) -> UnifiedTreeNode:
        if (
            self._is_fluxon_hostless_full_mode()
            and child.id in self.ongoing_fluxon_hostless_backup
        ):
            self._wait_for_fluxon_hostless_backup(child)
        new_node = UnifiedTreeNode(self.tree_components, priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count
        new_node.creation_time = child.creation_time

        self._for_each_component_lru(child, UnifiedLRUList.remove_node)

        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        for component in self._components_tuple:
            component.redistribute_on_node_split(new_parent=new_node, child=child)
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        # Both resulting Full-KV slices are immutable.  Refresh their events
        # after redistribution so a later parent-first Fluxon backup does not
        # depend on an event for the pre-split tensor layout.
        self._record_fluxon_hostless_source_ready_event(new_node)
        self._record_fluxon_hostless_source_ready_event(child)

        self._for_each_component_lru(
            new_node, UnifiedLRUList.insert_mru, skip_existing=True
        )
        self._for_each_component_lru(
            child, UnifiedLRUList.insert_mru, skip_existing=True
        )
        child.last_access_time = get_and_increase_time_counter()

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(child)
        return new_node

    def _touch_node(self, node: UnifiedTreeNode):
        node.last_access_time = get_and_increase_time_counter()
        if node != self.root_node:
            for comp in self._components_tuple:
                if comp.component_type == BASE_COMPONENT_TYPE:
                    continue
                comp.refresh_lru(LRURefreshPhase.WALKDOWN, node, self.root_node)

    def _add_new_node(
        self,
        parent: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        priority: int = 0,
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components, priority=priority)
        new_node.parent = parent
        new_node.key = key
        new_node.component_data[BASE_COMPONENT_TYPE].value = value.clone()
        self._record_fluxon_hostless_source_ready_event(new_node)
        parent.children[key.child_key(self.page_size)] = new_node
        self.component_evictable_size_[BASE_COMPONENT_TYPE] += len(value)
        if self.enable_storage:
            new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(parent)
        self._record_store_event(new_node)
        return new_node

    def _unevict_node_on_insert(
        self, node: UnifiedTreeNode, fresh_value: torch.Tensor
    ) -> None:
        """Restore an evicted node's Full device value from fresh KV indices
        during insert."""
        ct = BASE_COMPONENT_TYPE
        cd = node.component_data[ct]
        assert cd.value is None
        n = len(fresh_value)
        cd.value = fresh_value.clone()
        self._record_fluxon_hostless_source_ready_event(node)
        self.component_evictable_size_[ct] += n
        self._update_evictable_leaf_sets(node)
        if node.parent is not None:
            self._update_evictable_leaf_sets(node.parent)
        self._record_store_event(node, medium=StorageMedium.GPU)

    def _insert_helper(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        params: InsertParams,
    ) -> InsertResult:
        priority = params.priority
        if priority is None:
            priority = 0
        self._touch_node(node)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        child_key = key.child_key(self.page_size)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)
            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)
            node.priority = max(node.priority, priority)

            if node.evicted:
                self._unevict_node_on_insert(node, value[:prefix_len])
                # FULL was restored from the request's fresh KV. Aux
                # components (e.g. SWA) may still hold tombstones and need
                # to rebuild their value from the same slice.
                for component in self._components_tuple:
                    if component.component_type == BASE_COMPONENT_TYPE:
                        continue
                    component.recover_after_unevict(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        params=params,
                    )
            else:
                value_slice = value[:prefix_len]
                consumed_from = prefix_len
                # Let each component claim ownership of overlapping KV slots
                for component in self._components_tuple:
                    comp_consumed_from = component.update_component_on_insert_overlap(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        value_slice=value_slice,
                        params=params,
                    )
                    consumed_from = min(consumed_from, comp_consumed_from)

                dup_start = max(0, params.prev_prefix_len - total_prefix_length)
                if dup_start < consumed_from:
                    self.token_to_kv_pool_allocator.free(
                        value_slice[dup_start:consumed_from]
                    )

            self._inc_hit_count(node, params.chunked)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

        is_new_leaf = False
        # Create new leaf for remaining suffix
        if len(key):
            if any(
                comp.should_skip_leaf_creation(
                    total_prefix_len=total_prefix_length,
                    key_len=len(key),
                    params=params,
                )
                for comp in self._components_tuple
            ):
                # TODO: When leaf creation is skipped, We should release all component
                # resources here or propagate a flag so that
                # cleanup_after_caching_req can free them properly.
                self.token_to_kv_pool_allocator.free(value)
                return InsertResult(prefix_len=total_prefix_length)
            target_node = self._add_new_node(node, key, value, priority=priority)
            is_new_leaf = True
        else:
            target_node = node

        # Finalize: let each component attach its data to the target node.
        # e.g. Mamba attaches mamba_value to the leaf node
        result = InsertResult(prefix_len=total_prefix_length)
        for component in self._components_tuple:
            component.commit_insert_component_data(
                node=target_node,
                is_new_leaf=is_new_leaf,
                params=params,
                result=result,
            )

        if target_node is not self.root_node:
            for component in self._components_tuple:
                if component.component_type == BASE_COMPONENT_TYPE:
                    continue
                component.refresh_lru(
                    LRURefreshPhase.INSERT_END, target_node, self.root_node
                )

        if is_new_leaf:
            self._inc_hit_count(target_node, params.chunked)
        return result

    def _insert_helper_host(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        host_value: torch.Tensor,
        hash_value: list[str],
    ) -> InsertResult:
        total_len = len(key)
        self._touch_node(node)
        if total_len == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        child_key = key.child_key(self.page_size)
        matched_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)

            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)

            if len(key):
                child_key = key.child_key(self.page_size)

        result = InsertResult(
            prefix_len=matched_length,
        )
        if len(key) == 0:
            if (
                node is not self.root_node
                and node.component_data[BASE_COMPONENT_TYPE].host_value is not None
            ):
                result.inserted_host_node = node
            return result

        new_node = UnifiedTreeNode(self.tree_components, priority=node.priority)
        new_node.parent = node
        new_node.key = key
        new_node.hash_value = hash_value
        new_node.component_data[BASE_COMPONENT_TYPE].host_value = host_value.clone()
        node.children[child_key] = new_node
        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(node)
        result.inserted_host_node = new_node
        return result

    # ---- Evict Helpers ----

    def _cascade_evict(
        self,
        node: UnifiedTreeNode,
        trigger: TreeComponent,
        tracker: dict[ComponentType, int],
        target: EvictLayer = EvictLayer.DEVICE,
    ):
        """Cascade eviction from trigger to lower-or-equal priority components."""

        is_leaf = False
        if target == EvictLayer.DEVICE:
            is_leaf = node in self.evictable_device_leaves
        elif target == EvictLayer.HOST:
            is_leaf = node in self.evictable_host_leaves

        trigger_priority = trigger.eviction_priority(is_leaf)

        for comp in self._components_tuple:
            if comp.eviction_priority(is_leaf) <= trigger_priority:
                if comp is not trigger and comp.node_has_component_data(node, target):
                    cd = node.component_data[comp.component_type]
                    if EvictLayer.DEVICE in target:
                        assert cd.lock_ref == 0
                    if EvictLayer.HOST in target:
                        assert cd.host_lock_ref == 0
                    self._evict_component_and_detach_lru(
                        node, comp, target=target, tracker=tracker
                    )

        # Now that all components (including SWA which depends on Full.value)
        # have been freed, we can safely tombstone Full.value.
        # This is deferred from evict_component because free_swa needs it.
        if (
            target is EvictLayer.DEVICE
            and trigger.component_type == BASE_COMPONENT_TYPE
        ):
            node.component_data[trigger.component_type].value = None

        self._update_evictable_leaf_sets(node)

    def _remove_leaf_from_parent(self, node: UnifiedTreeNode):
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node

    def _evict_component_and_detach_lru(
        self,
        node: UnifiedTreeNode,
        comp: TreeComponent,
        target: EvictLayer = EvictLayer.DEVICE,
        tracker: dict[ComponentType, int] = None,
    ) -> tuple[int, int]:
        device_freed, host_freed = comp.evict_component(node, target=target)
        if tracker is not None:
            if EvictLayer.DEVICE in target:
                tracker[comp.component_type] += device_freed
            elif EvictLayer.HOST in target:
                tracker[comp.component_type] += host_freed

        # Detach from the appropriate LRU list(s)
        ct = comp.component_type
        for layer, lru_lists in (
            (EvictLayer.DEVICE, self.lru_lists),
            (EvictLayer.HOST, self.host_lru_lists),
        ):
            if layer in target:
                lru = lru_lists[ct]
                if lru.in_list(node):
                    lru.remove_node(node)
        return device_freed, host_freed

    def _iteratively_delete_tombstone_leaf(
        self, deleted_node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ):
        """Walk up from *deleted_node* and cascade-delete childless ancestors.

        Only the Full (base) component decides whether a node survives:
          - Full device present  → keep as D-leaf
          - Full host present    → keep as H-leaf
          - neither              → evict all remaining data, delete, continue up
        """
        ct = BASE_COMPONENT_TYPE
        cur = deleted_node.parent
        while cur != self.root_node and len(cur.children) == 0:
            if any(
                cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in cur.component_data
            ):
                break

            has_device = cur.component_data[ct].value is not None
            has_lower = (
                cur.component_data[ct].host_value is not None
                or self._is_component_storage_backed(cur.component_data[ct])
            )

            if has_device:
                self._update_evictable_leaf_sets(cur)
                break

            # Full device absent — clean up orphaned aux device data.
            for comp in self.components.values():
                if comp.node_has_component_data(cur):
                    self._evict_component_and_detach_lru(
                        cur, comp, target=EvictLayer.DEVICE, tracker=tracker
                    )

            if has_lower:
                self._update_evictable_leaf_sets(cur)
                break

            # Full absent on both layers — evict remaining host data, delete.
            for comp in self.components.values():
                if comp.node_has_component_data(cur, target=EvictLayer.HOST):
                    self._evict_component_and_detach_lru(
                        cur, comp, target=EvictLayer.HOST, tracker=tracker
                    )

            self.evictable_host_leaves.discard(cur)
            self._remove_leaf_from_parent(cur)
            parent = cur.parent
            self._update_evictable_leaf_sets(parent)
            cur = parent

    def _for_each_component_lru(
        self,
        node: UnifiedTreeNode,
        lru_op,
        target: EvictLayer = EvictLayer.DEVICE,
        skip_existing: bool = False,
    ):
        """Apply lru_op to each aux component's LRU that has data on this node.
        If skip_existing=True, skip components already in the target LRU list."""
        lru_dict = self.host_lru_lists if target is EvictLayer.HOST else self.lru_lists
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue  # Full uses leaf sets, not LRU
            cd = node.component_data[ct]
            if (cd.host_value if target is EvictLayer.HOST else cd.value) is not None:
                lru = lru_dict[ct]
                if skip_existing and lru.in_list(node):
                    continue
                lru_op(lru, node)

    def evict_host(
        self, num_tokens: int, component_type: ComponentType = BASE_COMPONENT_TYPE
    ) -> int:
        """Evict host resources for a specific component to free host pool space."""
        tracker: dict[ComponentType, int] = {ct: 0 for ct in self.tree_components}
        comp = self.components.get(component_type)
        if comp is not None:
            comp.drive_host_eviction(num_tokens, tracker)
        evicted = tracker[component_type]
        if evicted > 0:
            logger.info(
                "HiCache host evict result: requested=%d evicted=%d component=%s",
                num_tokens,
                evicted,
                component_type,
            )
        return tracker[component_type]

    def _is_device_leaf(self, node: UnifiedTreeNode) -> bool:
        """D-leaf: Full device value present, no child with Full KV on device,
        unlocked, not root.

        Only the Full (base) component is required; auxiliary components
        (Mamba, SWA) are not mandatory for D-leaf membership."""
        ct = BASE_COMPONENT_TYPE
        if node is self.root_node or node.evicted:
            return False
        if any(cd.lock_ref > 0 for cd in node.component_data):
            return False
        if any(
            child.component_data[ct].value is not None
            for child in node.children.values()
        ):
            return False
        return True

    def _is_host_leaf(self, node: UnifiedTreeNode) -> bool:
        """H-leaf: evicted, Full host value present, no children, unlocked, not root.

        Only the Full (base) component host_value is required; auxiliary
        components are not mandatory for H-leaf membership."""
        if node is self.root_node or not node.evicted:
            return False
        if node.component_data[BASE_COMPONENT_TYPE].host_value is None:
            return False
        if any(cd.host_lock_ref > 0 for cd in node.component_data):
            return False
        if len(node.children) > 0:
            return False
        return True

    def _update_evictable_leaf_sets(self, node: UnifiedTreeNode) -> None:
        """Update both device and host leaf sets for a node."""
        is_device_evictable = self._is_device_leaf(node)
        if is_device_evictable:
            self.evictable_device_leaves.add(node)
        else:
            self.evictable_device_leaves.discard(node)

        if self._is_host_leaf(node):
            self.evictable_host_leaves.add(node)
        else:
            self.evictable_host_leaves.discard(node)

    def _evict_to_host(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int] = None
    ) -> None:
        """GPU→CPU demotion: release all device resources, node stays in tree."""
        assert not node.evicted and node.backuped
        trigger = self.components[BASE_COMPONENT_TYPE]
        self._evict_component_and_detach_lru(
            node, trigger, target=EvictLayer.DEVICE, tracker=tracker
        )
        self._cascade_evict(node, trigger, tracker)
        self._record_remove_event(node, medium=StorageMedium.GPU)

        # after device eviction, insert aux components into host LRU.
        self._for_each_component_lru(
            node, UnifiedLRUList.insert_mru, target=EvictLayer.HOST, skip_existing=True
        )
        self._update_evictable_leaf_sets(node.parent)

    def _evict_to_fluxon_storage(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int] = None
    ) -> None:
        """Release device resources for a Fluxon-backed node without using host L2."""
        assert self._is_fluxon_hostless_full_mode()
        assert not node.evicted and node.backuped
        evict_start = time.perf_counter()
        base_value = node.component_data[BASE_COMPONENT_TYPE].value
        token_count = 0 if base_value is None else len(base_value)
        trigger = self.components[BASE_COMPONENT_TYPE]
        self._evict_component_and_detach_lru(
            node, trigger, target=EvictLayer.DEVICE, tracker=tracker
        )
        self._cascade_evict(node, trigger, tracker)
        self._record_remove_event(node, medium=StorageMedium.GPU)
        self._update_evictable_leaf_sets(node.parent)
        logger.debug(
            "HiCache fluxon evict success: node=%d tokens=%d evict_to_fluxon_ms=%.3f",
            node.id,
            token_count,
            (time.perf_counter() - evict_start) * 1000.0,
        )

    def _evict_device_leaf(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict a device leaf node, choosing the right strategy:

        - backuped: demote to host via _evict_to_host (node stays in tree)
        - not backuped + write_back: write_backup first, then demote
        - not backuped + write_through: Cascade evict all components

        All freed device tokens are accumulated into *tracker*.
        """
        assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"
        eviction_observation = self._active_fluxon_hostless_eviction_observation()
        base_value = node.component_data[BASE_COMPONENT_TYPE].value
        token_count = 0 if base_value is None else len(base_value)
        if eviction_observation is not None:
            eviction_observation["evict_candidate_tokens"] += token_count
        hostless_storage_ready = (
            self._fluxon_hostless_node_storage_ready(node)
            if self._is_fluxon_hostless_full_mode()
            else False
        )
        if self._is_fluxon_hostless_full_mode():
            if not hostless_storage_ready:
                if (
                    self.cache_controller is not None
                    and self.cache_controller.write_policy == "write_back"
                ):
                    had_pending_writeback = (
                        node.id in self.ongoing_fluxon_hostless_backup
                        or self._has_pending_fluxon_hostless_ack(node.id)
                    )
                    if eviction_observation is not None:
                        counter = (
                            "evict_pending_writebacks"
                            if had_pending_writeback
                            else "evict_new_writebacks"
                        )
                        eviction_observation[counter] += 1
                    write_backup_started_at = time.perf_counter()
                    write_backup_tokens = self.write_backup(node, write_back=True)
                    if eviction_observation is not None:
                        eviction_observation["evict_write_backup_ms"] += (
                            time.perf_counter() - write_backup_started_at
                        ) * 1000.0
                    if write_backup_tokens <= 0:
                        return
                    write_wait_started_at = time.perf_counter()
                    self.writing_check(write_back=True)
                    if eviction_observation is not None:
                        eviction_observation["evict_write_wait_ms"] += (
                            time.perf_counter() - write_wait_started_at
                        ) * 1000.0
                    hostless_storage_ready = self._fluxon_hostless_node_storage_ready(
                        node
                    )
                    if not hostless_storage_ready:
                        return
                    if eviction_observation is not None:
                        eviction_observation[
                            "evict_after_writeback_tokens"
                        ] += token_count
                    self._evict_to_fluxon_storage(node, tracker)
                    return
                write_backup_started_at = time.perf_counter()
                write_through_recoverable = (
                    self._ensure_fluxon_hostless_write_through_recoverable(node)
                )
                if eviction_observation is not None:
                    eviction_observation["evict_write_backup_ms"] += (
                        time.perf_counter() - write_backup_started_at
                    ) * 1000.0
                if write_through_recoverable:
                    if eviction_observation is not None:
                        eviction_observation[
                            "evict_after_writeback_tokens"
                        ] += token_count
                    self._evict_to_fluxon_storage(node, tracker)
                    return
                # Write-through: node has no recoverable lower tier, delete entirely.
                if eviction_observation is not None:
                    eviction_observation[
                        "evict_unbacked_drop_tokens"
                    ] += token_count
                self._record_remove_event(node, medium=StorageMedium.GPU)
                for comp in self._components_tuple:
                    self._evict_component_and_detach_lru(
                        node, comp, target=EvictLayer.ALL, tracker=tracker
                    )
                self.evictable_device_leaves.discard(node)
                parent = node.parent
                self._remove_leaf_from_parent(node)
                self._update_evictable_leaf_sets(parent)
                self._iteratively_delete_tombstone_leaf(node, tracker)
                return
            if eviction_observation is not None:
                eviction_observation["evict_already_backed_tokens"] += token_count
            self._evict_to_fluxon_storage(node, tracker)
            return

        if not node.backuped or (
            self._is_fluxon_hostless_full_mode() and not hostless_storage_ready
        ):
            if (
                self.cache_controller is not None
                and self.cache_controller.write_policy == "write_back"
            ):
                if self.write_backup(node, write_back=True) <= 0:
                    return
                self.writing_check(write_back=True)
                if self._is_fluxon_hostless_full_mode():
                    self._evict_to_fluxon_storage(node, tracker)
                    return
                self._evict_to_host(node, tracker)
                return
            else:
                # Write-through: node has no backup, delete entirely.
                self._record_remove_event(node, medium=StorageMedium.GPU)
                for comp in self._components_tuple:
                    self._evict_component_and_detach_lru(
                        node, comp, target=EvictLayer.ALL, tracker=tracker
                    )
                self.evictable_device_leaves.discard(node)
                parent = node.parent
                self._remove_leaf_from_parent(node)
                self._update_evictable_leaf_sets(parent)
                self._iteratively_delete_tombstone_leaf(node, tracker)
                return
        if self._is_fluxon_hostless_full_mode():
            self._evict_to_fluxon_storage(node, tracker)
        else:
            self._evict_to_host(node, tracker)

    def _evict_host_leaf(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ) -> None:
        """Atomically evict all components on a host leaf.

        All freed tokens are accumulated into *tracker*."""
        assert self._is_host_leaf(node), f"node {node.id} is not an H-leaf"

        self._record_remove_event(node, medium=StorageMedium.CPU)
        for comp in self._components_tuple:
            _, hf = self._evict_component_and_detach_lru(
                node, comp, target=EvictLayer.ALL, tracker=None
            )
            tracker[comp.component_type] += hf
        self.evictable_host_leaves.discard(node)
        self._remove_leaf_from_parent(node)
        self._iteratively_delete_tombstone_leaf(node, tracker)

    # ---- HiCache: Backup / LoadBack ----

    def write_backup(self, node: UnifiedTreeNode, write_back: bool = False) -> int:
        """Backup a node's data from device to host (D->H)."""
        if self.cache_controller is None:
            return 0
        if self._is_fluxon_hostless_full_mode():
            base_value = node.component_data[BASE_COMPONENT_TYPE].value
            if self._fluxon_hostless_node_storage_ready(node):
                return 0 if base_value is None else len(base_value)

            if node.id in self.ongoing_fluxon_hostless_backup or self._has_pending_fluxon_hostless_ack(
                node.id
            ):
                return 0 if base_value is None else len(base_value)

            if (
                node.parent is not self.root_node
                and not self._fluxon_hostless_node_storage_ready(node.parent)
            ):
                if self.write_backup(node.parent, write_back=write_back) <= 0:
                    return 0
                if not self._fluxon_hostless_node_storage_ready(node.parent):
                    if write_back:
                        self._wait_for_fluxon_hostless_backup(node.parent)
                    if not self._fluxon_hostless_node_storage_ready(node.parent):
                        return 0

            backend = self._fluxon_backend()
            if backend is None:
                return 0
            device_value = base_value
            if device_value is None or len(device_value) == 0:
                return 0
            try:
                batch = self._submit_fluxon_hostless_write_batch(
                    node, backend, write_back=write_back
                )
            except Exception as err:
                logger.warning(
                    "HiCache write_backup submit failed: node=%d mode=fluxon_hostless_full error=%s",
                    node.id,
                    err,
                )
                return 0
            logger.info(
                "HiCache write_backup submitted: node=%d tokens=%d mode=fluxon_hostless_full "
                "write_back=%s dedicated_write_stream=%s storage_pending=%s "
                "storage_staged=%s",
                node.id,
                len(device_value),
                write_back,
                batch.dedicated_write_stream,
                bool(
                    node.component_data[BASE_COMPONENT_TYPE].metadata.get(
                        "storage_pending", False
                    )
                ),
                bool(
                    node.component_data[BASE_COMPONENT_TYPE].metadata.get(
                        "storage_staged", False
                    )
                ),
            )
            return len(device_value)

        # Backup invariant (write-through): parent must be backuped first
        if not write_back and (
            node.parent is not self.root_node and not node.parent.backuped
        ):
            if self.write_backup(node.parent) <= 0:
                return 0

        device_value = node.component_data[BASE_COMPONENT_TYPE].value
        kv_xfer = PoolTransfer(name=PoolName.KV, device_indices=device_value)

        # Build aux transfers, keyed per component.
        comp_xfers: dict[ComponentType, list] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            t = comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_HOST)
            if t:
                comp_xfers[comp.component_type] = t
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_HOST, kv_xfer, comp_xfers
        )

        # Pre-evict host if insufficient
        kv_tokens = len(device_value)
        host_avail = self.cache_controller.mem_pool_host.available_size()
        if host_avail < kv_tokens:
            needed = kv_tokens - host_avail
            evicted = self.evict_host(needed)
            if evicted < needed:
                return 0

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        host_indices = self.cache_controller.write(
            device_value, node_id=node.id, extra_pools=aux_xfers or None
        )
        if host_indices is None:
            return 0

        # Commit
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            node,
            CacheTransferPhase.BACKUP_HOST,
            transfers=[kv_xfer],
        )
        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                node,
                CacheTransferPhase.BACKUP_HOST,
                transfers=xfers,
            )

        lock_params = None
        if not write_back:
            lock_params = self.inc_lock_ref(node).to_dec_params()
        self.ongoing_write_through[node.id] = (node, lock_params)
        logger.info(
            "HiCache write_backup: node=%d tokens=%d mode=hicache_host write_back=%s aux_transfers=%d",
            node.id,
            len(host_indices),
            write_back,
            len(aux_xfers),
        )
        return len(host_indices)

    def load_back(
        self,
        best_match_node: UnifiedTreeNode,
        mem_quota: Optional[int] = None,
        req=None,
    ) -> bool:
        """Load evicted KV data from host back to device (H→D)."""
        if self.cache_controller is None:
            return False
        start_time = time.perf_counter()
        if self._is_fluxon_hostless_full_mode():
            req_id = str(getattr(req, "rid", "")) if req is not None else ""
            backend = self._fluxon_backend()
            if backend is None:
                if req_id:
                    self._finish_fluxon_hostless_request_observation(
                        req_id,
                        "load_back_backend_unavailable",
                        anchor_node_id=best_match_node.id,
                        host_hit_tokens=int(getattr(req, "host_hit_length", 0)),
                    )
                return False

            nodes_to_load: list[UnifiedTreeNode] = []
            cursor = best_match_node
            kv_tokens = 0
            while cursor is not self.root_node and cursor.evicted:
                if not cursor.backuped:
                    break
                nodes_to_load.append(cursor)
                kv_tokens += len(cursor.key)
                cursor = cursor.parent

            mamba_restore_node = (
                getattr(req, "mamba_state_node", None) if req is not None else None
            ) or best_match_node
            mamba_needs_restore = self._hostless_mamba_needs_restore(
                mamba_restore_node
            )
            if not nodes_to_load and not mamba_needs_restore:
                if req_id:
                    self._finish_fluxon_hostless_request_observation(
                        req_id,
                        "load_back_nothing_to_restore",
                        anchor_node_id=best_match_node.id,
                        host_hit_tokens=int(getattr(req, "host_hit_length", 0)),
                    )
                return False

            lock_result = self.inc_lock_ref(best_match_node)
            ancestor_lock_params = lock_result.to_dec_params()
            mamba_lock_params = None
            if mamba_needs_restore and mamba_restore_node is not best_match_node:
                mamba_lock_params = self.inc_lock_ref(
                    mamba_restore_node
                ).to_dec_params()
            restored_nodes: list[tuple[UnifiedTreeNode, torch.Tensor]] = []
            restored_mamba: Optional[tuple[UnifiedTreeNode, torch.Tensor]] = None
            alloc_ms = 0.0
            evict_ms = 0.0
            pin_ms = 0.0
            batch_get_ms = 0.0
            restore_kv_ms = 0.0
            restore_mamba_ms = 0.0
            restore_sync_ms = 0.0
            last_read_node_id: int | None = None
            last_read_pages = 0
            last_read_first_key = ""
            last_read_last_key = ""
            last_read_key_sig = ""
            last_recoverable_error_kind = "none"
            raw_h2d_state = _FluxonRawH2DSubmitState(self)
            hostless_plan = self._fluxon_hostless_plan()
            if req_id and req_id in self._fluxon_hostless_request_observations:
                observation = self._fluxon_hostless_request_observations[req_id]
                observation["ready_bytes"] = int(
                    observation.get("ready_pages", 0)
                ) * int(hostless_plan["total_bytes"])
            if not bool(hostless_plan["page_value_path"]):
                raise RuntimeError("Fluxon hostless load_back requires page-value restore")
            mamba_state_key: str | None = None
            kv_batch_hash_values: list[str] = []
            kv_batch_atomic_group_lens: list[int] = []
            kv_batch_nodes: list[UnifiedTreeNode] = []
            kv_plan_ptr: int | None = None
            kv_plan_offset_pages = 0
            kv_gpu_staging_lease = None
            mamba_plan_ptr: int | None = None
            ready_prefetch_operation: _FluxonHostlessPrefetchOperation | None = None
            eviction_observation = self._new_fluxon_hostless_eviction_observation(0)
            queued_page_indices: torch.Tensor | None = None
            defer_ancestor_unlock = False
            layerwise_async = kv_tokens > 0 and not mamba_needs_restore
            try:
                if kv_tokens < self.load_back_threshold and not mamba_needs_restore:
                    if req_id:
                        self._finish_fluxon_hostless_request_observation(
                            req_id,
                            "load_back_below_threshold",
                            anchor_node_id=best_match_node.id,
                            host_hit_tokens=int(
                                getattr(req, "host_hit_length", kv_tokens)
                            ),
                        )
                    return False
                if (
                    kv_tokens > 0
                    and mem_quota is not None
                    and kv_tokens > mem_quota + lock_result.delta
                    and not mamba_needs_restore
                ):
                    if req_id:
                        self._finish_fluxon_hostless_request_observation(
                            req_id,
                            "load_back_over_mem_quota",
                            anchor_node_id=best_match_node.id,
                            host_hit_tokens=int(
                                getattr(req, "host_hit_length", kv_tokens)
                            ),
                        )
                    return False

                if req is not None:
                    ready_prefetch_operation = self.fluxon_hostless_ready_prefetch.pop(
                        req.rid,
                        None,
                    )
                if ready_prefetch_operation is None:
                    last_recoverable_error_kind = "not_ready"
                    raise _FluxonHostlessCacheMiss(
                        "Fluxon hostless load_back requires a ready restore entry: "
                        f"best_match_node={best_match_node.id}"
                    )
                if req_id:
                    self._observe_fluxon_hostless_request(
                        req_id,
                        load_back_consume_start_age_ms=(
                            self._fluxon_hostless_observation_age_ms(req_id)
                        ),
                    )

                if kv_tokens > 0:
                    kv_batch_nodes = list(reversed(nodes_to_load))
                    for node in kv_batch_nodes:
                        hash_values = self._node_hash_values(node)
                        if len(hash_values) == 0:
                            raise RuntimeError(
                                "Fluxon hostless ready restore requires non-empty node hash values: "
                                f"node={node.id} tokens={len(node.key)}"
                            )
                        kv_batch_hash_values.extend(hash_values)
                        kv_batch_atomic_group_lens.append(len(hash_values))
                    (
                        last_read_pages,
                        last_read_first_key,
                        last_read_last_key,
                        last_read_key_sig,
                    ) = _fluxon_key_signature(kv_batch_hash_values)
                    last_read_node_id = best_match_node.id
                    if self._fluxon_hostless_ready_operation_covers_kv(
                        ready_prefetch_operation,
                        kv_batch_hash_values,
                        exact=False,
                    ):
                        kv_plan_offset_pages = int(
                            ready_prefetch_operation.kv_plan_offset_pages
                        )
                        kv_plan_ptr = ready_prefetch_operation.kv_plan_ptr
                        ready_prefetch_operation.kv_plan_ptr = None
                        if kv_plan_ptr is None:
                            raise RuntimeError(
                                "Fluxon ready restore lost its ordered source plan"
                            )
                        if ready_prefetch_operation.gpu_direct:
                            staging_lease = (
                                ready_prefetch_operation.gpu_staging_lease
                            )
                            if staging_lease is None:
                                raise RuntimeError(
                                    "Fluxon ready GPU restore lost its staging lease"
                                )
                            remote_pages = sum(
                                kv_plan_offset_pages
                                <= index
                                < kv_plan_offset_pages + len(kv_batch_hash_values)
                                for index in ready_prefetch_operation.gpu_remote_indices
                            )
                            if kv_plan_offset_pages == 0:
                                staging_lease.trim_after_transfer(remote_pages)
                            kv_gpu_staging_lease = staging_lease
                            ready_prefetch_operation.gpu_staging_lease = None
                    else:
                        last_recoverable_error_kind = "key_missing"
                        raise _FluxonHostlessCacheMiss(
                            "Fluxon hostless ready restore entry does not cover KV batch: "
                            f"best_match_node={best_match_node.id} "
                            f"expected={len(kv_batch_hash_values)} "
                            f"ready_tokens={ready_prefetch_operation.completed_tokens} "
                            f"anchor_node={ready_prefetch_operation.anchor_node_id}"
                        )

                if mamba_needs_restore:
                    mamba_state_key = self._node_mamba_storage_key(mamba_restore_node)
                    if mamba_state_key is None:
                        raise RuntimeError(
                            "Hostless Fluxon Mamba restore requires node hash: "
                            f"node={mamba_restore_node.id}"
                        )
                    if (
                        ready_prefetch_operation.mamba_anchor_node_id
                        == mamba_restore_node.id
                        and ready_prefetch_operation.mamba_key == mamba_state_key
                        and ready_prefetch_operation.mamba_plan_ptr is not None
                    ):
                        mamba_plan_ptr = ready_prefetch_operation.mamba_plan_ptr
                        ready_prefetch_operation.mamba_plan_ptr = None
                    else:
                        last_recoverable_error_kind = "key_missing"
                        last_read_node_id = mamba_restore_node.id
                        last_read_pages = 1
                        last_read_first_key = mamba_state_key
                        last_read_last_key = mamba_state_key
                        last_read_key_sig = _fluxon_key_signature([mamba_state_key])[3]
                        raise _FluxonHostlessCacheMiss(
                            "Fluxon hostless ready restore entry does not cover Mamba state for "
                            f"node={mamba_restore_node.id}"
                        )

                if kv_tokens > 0:
                    if self.supports_swa():
                        avail = self.token_to_kv_pool_allocator.full_available_size()
                    else:
                        avail = self.token_to_kv_pool_allocator.available_size()
                    if avail < kv_tokens:
                        needed = kv_tokens - avail
                        eviction_observation = (
                            self._new_fluxon_hostless_eviction_observation(needed)
                        )
                        self._fluxon_hostless_eviction_observation_stack.append(
                            eviction_observation
                        )
                        evict_start = time.perf_counter()
                        try:
                            evict_result = self.evict(EvictParams(num_tokens=needed))
                        finally:
                            evict_ms += (
                                time.perf_counter() - evict_start
                            ) * 1000.0
                            eviction_observation["evict_total_ms"] = evict_ms
                            if (
                                self._fluxon_hostless_eviction_observation_stack
                                and self._fluxon_hostless_eviction_observation_stack[-1]
                                is eviction_observation
                            ):
                                self._fluxon_hostless_eviction_observation_stack.pop()
                            else:
                                logger.warning(
                                    "Fluxon hostless eviction observation stack mismatch: "
                                    "req=%s node=%d",
                                    req_id,
                                    best_match_node.id,
                                )
                        eviction_observation["evict_actual_tokens"] = int(
                            evict_result.num_tokens_evicted
                        )
                        if evict_result.num_tokens_evicted < needed:
                            if kv_plan_ptr is not None:
                                backend.release_views(kv_plan_ptr)
                                kv_plan_ptr = None
                            if kv_gpu_staging_lease is not None:
                                self._release_fluxon_gpu_staging_lease(
                                    kv_gpu_staging_lease,
                                    "load_back_evict_insufficient",
                                )
                                kv_gpu_staging_lease = None
                            if mamba_plan_ptr is not None:
                                backend.release_views(mamba_plan_ptr)
                                mamba_plan_ptr = None
                            if req_id:
                                self._finish_fluxon_hostless_request_observation(
                                    req_id,
                                    "load_back_evict_insufficient",
                                    anchor_node_id=best_match_node.id,
                                    host_hit_tokens=int(
                                        getattr(req, "host_hit_length", kv_tokens)
                                    ),
                                    **eviction_observation,
                                )
                            return False

                    allocated_node_indices: list[
                        tuple[UnifiedTreeNode, torch.Tensor]
                    ] = []
                    uncommitted_node_indices: list[
                        tuple[UnifiedTreeNode, torch.Tensor]
                    ] = []
                    try:
                        if len(kv_batch_hash_values) == 0:
                            raise RuntimeError(
                                "Fluxon hostless ready restore requires prepared batch keys"
                            )
                        if len(kv_batch_nodes) == 0:
                            raise RuntimeError(
                                "Fluxon hostless ready restore requires prepared batch nodes"
                            )
                        first_cd = kv_batch_nodes[0].component_data[
                            BASE_COMPONENT_TYPE
                        ]
                        logger.info(
                            "Fluxon hostless read keys: best_match_node=%d "
                            "nodes=%d tokens=%d evicted=%s backuped=%s "
                            "storage_staged=%s storage_backed=%s pages=%d "
                            "groups=%s first_key=%s last_key=%s key_sig=%s "
                            "trigger=%s scheduler_reason=%s",
                            best_match_node.id,
                            len(kv_batch_nodes),
                            kv_tokens,
                            best_match_node.evicted,
                            best_match_node.backuped,
                            bool(first_cd.metadata.get("storage_staged", False)),
                            bool(first_cd.metadata.get("storage_backed", False)),
                            last_read_pages,
                            kv_batch_atomic_group_lens,
                            last_read_first_key,
                            last_read_last_key,
                            last_read_key_sig,
                            getattr(
                                req,
                                "hicache_load_back_trigger",
                                "unknown_host_hit_match",
                            ),
                            getattr(
                                req,
                                "hicache_load_back_scheduler_reason",
                                "unknown_scheduler_reason",
                            ),
                        )
                        page_index_parts: list[torch.Tensor] = []
                        for node, expected_pages in zip(
                            kv_batch_nodes, kv_batch_atomic_group_lens
                        ):
                            alloc_start = time.perf_counter()
                            device_indices = self.token_to_kv_pool_allocator.alloc(
                                len(node.key)
                            )
                            alloc_ms += (
                                time.perf_counter() - alloc_start
                            ) * 1000.0
                            if device_indices is None:
                                for (
                                    _node,
                                    node_indices,
                                ) in uncommitted_node_indices:
                                    self.token_to_kv_pool_allocator.free(node_indices)
                                uncommitted_node_indices.clear()
                                if kv_plan_ptr is not None:
                                    backend.release_views(kv_plan_ptr)
                                    kv_plan_ptr = None
                                if kv_gpu_staging_lease is not None:
                                    self._release_fluxon_gpu_staging_lease(
                                        kv_gpu_staging_lease,
                                        "load_back_alloc_failed",
                                    )
                                    kv_gpu_staging_lease = None
                                if mamba_plan_ptr is not None:
                                    backend.release_views(mamba_plan_ptr)
                                    mamba_plan_ptr = None
                                if req_id:
                                    self._finish_fluxon_hostless_request_observation(
                                        req_id,
                                        "load_back_alloc_failed",
                                        anchor_node_id=best_match_node.id,
                                        host_hit_tokens=int(
                                            getattr(req, "host_hit_length", kv_tokens)
                                        ),
                                        **eviction_observation,
                                    )
                                return False
                            allocated_node_indices.append((node, device_indices))
                            uncommitted_node_indices.append((node, device_indices))
                            page_index_parts.append(
                                self._hostless_page_indices(
                                    device_indices,
                                    expected_pages=expected_pages,
                                )
                            )
                        page_indices = torch.cat(page_index_parts)
                        if kv_plan_ptr is None:
                            raise RuntimeError(
                                "Fluxon hostless KV restore requires a ready source"
                            )
                        restore_start = time.perf_counter()
                        if layerwise_async:
                            queued_page_indices = page_indices
                        else:
                            ordered_value_ptrs = (
                                self._fluxon_hostless_ready_value_ptrs(
                                    backend,
                                    kv_plan_ptr,
                                    ready_prefetch_operation,
                                    int(page_indices.numel()),
                                )
                            )
                            ordered_plan_blob = torch.empty(
                                len(ordered_value_ptrs) + 2,
                                dtype=torch.int64,
                                pin_memory=True,
                            )
                            ordered_plan_blob[0] = _FLUXON_PLAN_BLOB_MAGIC
                            ordered_plan_blob[1] = len(ordered_value_ptrs)
                            ordered_plan_blob[2:] = torch.tensor(
                                ordered_value_ptrs,
                                dtype=torch.int64,
                            )
                            restore_plan_ptr = int(ordered_plan_blob.data_ptr())
                            raw_h2d_state.retain(ordered_plan_blob)
                            if kv_gpu_staging_lease is not None:
                                raw_h2d_state.add_finalizer(
                                    lambda lease=kv_gpu_staging_lease: lease.release(
                                        "sync_restore_finalizer"
                                    )
                                )
                                kv_gpu_staging_lease = None
                                raw_h2d_state.mark_pending()
                            if restore_plan_ptr is None:
                                raise RuntimeError(
                                    "Fluxon hostless synchronous restore lost its source"
                                )
                            try:
                                if hostless_plan["is_mla"]:
                                    restore_mla_pages_from_fluxon_values(
                                        restore_plan_ptr,
                                        page_indices,
                                        hostless_plan["layer_ptrs"],
                                        int(hostless_plan["page_bytes"]),
                                        self._cuda_device_index(),
                                    )
                                else:
                                    restore_mha_pages_from_fluxon_values(
                                        restore_plan_ptr,
                                        page_indices,
                                        hostless_plan["k_layer_ptrs"],
                                        hostless_plan["v_layer_ptrs"],
                                        int(hostless_plan["k_page_bytes"]),
                                        int(hostless_plan["v_page_bytes"]),
                                        self._cuda_device_index(),
                                    )
                            except Exception:
                                if kv_plan_ptr is not None:
                                    backend.release_views(kv_plan_ptr)
                                    kv_plan_ptr = None
                                raise
                            if kv_plan_ptr is not None:
                                raw_h2d_state.add_finalizer(
                                    lambda plan_ptr=kv_plan_ptr, release=backend.release_views: release(
                                        plan_ptr
                                    )
                                )
                                kv_plan_ptr = None
                            raw_h2d_state.mark_pending()
                        restore_kv_ms += (
                            time.perf_counter() - restore_start
                        ) * 1000.0

                        assigned_tokens = sum(
                            len(device_indices)
                            for _, device_indices in allocated_node_indices
                        )
                        if assigned_tokens != kv_tokens:
                            raise RuntimeError(
                                "Fluxon hostless ready restore token count mismatch: "
                                f"assigned={assigned_tokens} expected={kv_tokens}"
                            )

                        for node, device_indices in allocated_node_indices:
                            node.component_data[BASE_COMPONENT_TYPE].value = (
                                device_indices.clone()
                            )
                            self.component_evictable_size_[BASE_COMPONENT_TYPE] += len(
                                device_indices
                            )
                            restored_nodes.append((node, device_indices))
                            uncommitted_node_indices.pop(0)
                            self._record_store_event(node, medium=StorageMedium.GPU)
                            self._update_evictable_leaf_sets(node)
                            if node.parent is not None:
                                self._update_evictable_leaf_sets(node.parent)
                    except Exception:
                        restore_sync_ms += raw_h2d_state.synchronize()
                        for _node, node_indices in uncommitted_node_indices:
                            self.token_to_kv_pool_allocator.free(node_indices)
                        raise

                if mamba_needs_restore:
                    mamba_start = time.perf_counter()
                    ready_mamba_plan_ptr = mamba_plan_ptr
                    mamba_plan_ptr = None
                    restored_mamba_indices = self._restore_hostless_mamba_state(
                        mamba_restore_node,
                        backend,
                        ready_mamba_plan_ptr,
                        raw_h2d_state,
                        req=req,
                    )
                    restore_mamba_ms = (time.perf_counter() - mamba_start) * 1000.0
                    restored_mamba = (mamba_restore_node, restored_mamba_indices)

                if not layerwise_async:
                    restore_sync_ms += raw_h2d_state.synchronize()

                self._update_evictable_leaf_sets(best_match_node)
                if mamba_restore_node is not best_match_node:
                    self._update_evictable_leaf_sets(mamba_restore_node)
                if self.metrics_collector is not None and kv_tokens > 0:
                    self.metrics_collector.observe_load_back_duration(
                        time.perf_counter() - start_time
                    )
                    self.metrics_collector.increment_load_back_num_tokens(kv_tokens)
                logger.info(
                    "init_load_back success: loaded %d tokens for node %d "
                    "(mode=%s restored_nodes=%d restored_mamba=%s mamba_node=%s) "
                    "alloc_ms=%.3f evict_ms=%.3f pin_ms=%.3f batch_get_ms=%.3f "
                    "restore_kv_ms=%.3f restore_mamba_ms=%.3f restore_sync_ms=%.3f duration_ms=%.3f",
                    kv_tokens,
                    best_match_node.id,
                    (
                        "fluxon_gpu_direct_d2d_layerwise"
                        if layerwise_async and kv_gpu_staging_lease is not None
                        else (
                            "fluxon_hostless_layerwise_async"
                            if layerwise_async
                            else "fluxon_hostless_full"
                        )
                    ),
                    len(restored_nodes),
                    restored_mamba is not None,
                    mamba_restore_node.id if mamba_needs_restore else None,
                    alloc_ms,
                    evict_ms,
                    pin_ms,
                    batch_get_ms,
                    restore_kv_ms,
                    restore_mamba_ms,
                    restore_sync_ms,
                    (time.perf_counter() - start_time) * 1000.0,
                )
                if layerwise_async:
                    if queued_page_indices is None or kv_plan_ptr is None:
                        raise RuntimeError(
                            "Fluxon layerwise restore is missing its queued source"
                        )
                    value_ptrs = self._fluxon_hostless_ready_value_ptrs(
                        backend,
                        kv_plan_ptr,
                        ready_prefetch_operation,
                        int(queued_page_indices.numel()),
                    )
                    operation = _FluxonHostlessLayerwiseLoad(
                        backend=backend,
                        plan_ptr=kv_plan_ptr,
                        value_ptrs=value_ptrs,
                        page_indices=queued_page_indices,
                        restore_plan=hostless_plan,
                        node_id=best_match_node.id,
                        req_id=req_id,
                        token_count=kv_tokens,
                        restored_nodes=list(restored_nodes),
                        gpu_staging_lease=kv_gpu_staging_lease,
                    )
                    if best_match_node.id in self.ongoing_load_back:
                        raise RuntimeError(
                            "Fluxon layerwise restore node is already in flight: "
                            f"node={best_match_node.id}"
                        )
                    self.fluxon_hostless_load_queue.append(operation)
                    self.ongoing_fluxon_hostless_layerwise_load[
                        best_match_node.id
                    ] = operation
                    self.ongoing_load_back[best_match_node.id] = (
                        best_match_node,
                        ancestor_lock_params,
                    )
                    defer_ancestor_unlock = True
                    kv_plan_ptr = None
                    kv_gpu_staging_lease = None
                    if req_id:
                        self._observe_fluxon_hostless_request(
                            req_id,
                            restore_queued_age_ms=(
                                self._fluxon_hostless_observation_age_ms(req_id)
                            ),
                        )
                if req_id:
                    consumed_fields = {
                        "anchor_node_id": best_match_node.id,
                        "host_hit_tokens": int(
                            getattr(req, "host_hit_length", kv_tokens)
                        ),
                        "consumed_pages": last_read_pages,
                        "consumed_tokens": kv_tokens,
                        "consumed_bytes": last_read_pages
                        * int(hostless_plan["total_bytes"]),
                        **eviction_observation,
                    }
                    if layerwise_async:
                        self._observe_fluxon_hostless_request(
                            req_id,
                            **consumed_fields,
                        )
                    else:
                        self._finish_fluxon_hostless_request_observation(
                            req_id,
                            "load_back_consumed",
                            **consumed_fields,
                        )
                return True
            except _FluxonHostlessCacheMiss as exc:
                restore_sync_ms += raw_h2d_state.synchronize()
                if restored_mamba is not None:
                    node, mamba_indices = restored_mamba
                    self._rollback_restored_hostless_mamba_state(node, mamba_indices)
                for node, device_indices in restored_nodes:
                    if node.component_data[BASE_COMPONENT_TYPE].value is not None:
                        node.component_data[BASE_COMPONENT_TYPE].value = None
                        self.component_evictable_size_[BASE_COMPONENT_TYPE] -= len(
                            device_indices
                        )
                        self._update_evictable_leaf_sets(node)
                        if node.parent is not None:
                            self._update_evictable_leaf_sets(node.parent)
                    self.token_to_kv_pool_allocator.free(device_indices)
                logger.warning(
                    "Fluxon hostless load_back cache miss: node=%d tokens=%d "
                    "restored_nodes=%d restored_mamba=%s batch_get_ms=%.3f "
                    "pin_ms=%.3f restore_sync_ms=%.3f duration_ms=%.3f "
                    "trigger=%s scheduler_reason=%s path=%s "
                    "recoverable_error_kind=%s read_node=%s read_pages=%d "
                    "read_first_key=%s read_last_key=%s read_key_sig=%s error=%s",
                    best_match_node.id,
                    kv_tokens,
                    len(restored_nodes),
                    restored_mamba is not None,
                    batch_get_ms,
                    pin_ms,
                    restore_sync_ms,
                    (time.perf_counter() - start_time) * 1000.0,
                    getattr(req, "hicache_load_back_trigger", "unknown_host_hit_match"),
                    getattr(
                        req,
                        "hicache_load_back_scheduler_reason",
                        "unknown_scheduler_reason",
                    ),
                    "match_prefix->schedule_policy->init_load_back->fluxon.ready_restore_entry",
                    last_recoverable_error_kind,
                    last_read_node_id,
                    last_read_pages,
                    last_read_first_key,
                    last_read_last_key,
                    last_read_key_sig,
                    exc,
                )
                if req_id:
                    self._finish_fluxon_hostless_request_observation(
                        req_id,
                        f"load_back_{last_recoverable_error_kind}",
                        anchor_node_id=best_match_node.id,
                        host_hit_tokens=int(
                            getattr(req, "host_hit_length", kv_tokens)
                        ),
                        consumed_pages=0,
                        consumed_tokens=0,
                        consumed_bytes=0,
                        **eviction_observation,
                    )
                return False
            except Exception as exc:
                restore_sync_ms += raw_h2d_state.synchronize()
                if restored_mamba is not None:
                    node, mamba_indices = restored_mamba
                    self._rollback_restored_hostless_mamba_state(node, mamba_indices)
                for node, device_indices in restored_nodes:
                    if node.component_data[BASE_COMPONENT_TYPE].value is not None:
                        node.component_data[BASE_COMPONENT_TYPE].value = None
                        self.component_evictable_size_[BASE_COMPONENT_TYPE] -= len(
                            device_indices
                        )
                        self._update_evictable_leaf_sets(node)
                        if node.parent is not None:
                            self._update_evictable_leaf_sets(node.parent)
                    self.token_to_kv_pool_allocator.free(device_indices)
                if req_id:
                    self._finish_fluxon_hostless_request_observation(
                        req_id,
                        "load_back_exception",
                        anchor_node_id=best_match_node.id,
                        host_hit_tokens=int(
                            getattr(req, "host_hit_length", kv_tokens)
                        ),
                        exception_type=type(exc).__name__,
                        **eviction_observation,
                    )
                raise
            finally:
                if kv_plan_ptr is not None:
                    backend.release_views(kv_plan_ptr)
                    kv_plan_ptr = None
                if kv_gpu_staging_lease is not None:
                    self._release_fluxon_gpu_staging_lease(
                        kv_gpu_staging_lease,
                        "load_back_finally",
                    )
                    kv_gpu_staging_lease = None
                if mamba_plan_ptr is not None:
                    backend.release_views(mamba_plan_ptr)
                    mamba_plan_ptr = None
                if ready_prefetch_operation is not None:
                    self._cancel_fluxon_hostless_prefetch_operation(
                        ready_prefetch_operation,
                        "load_back_ready_prefetch_finally",
                    )
                if mamba_lock_params is not None:
                    self.dec_lock_ref(mamba_restore_node, mamba_lock_params)
                if not defer_ancestor_unlock:
                    self.dec_lock_ref(best_match_node, ancestor_lock_params)

        # Build KV transfer
        kv_xfer = self.components[BASE_COMPONENT_TYPE].build_hicache_transfers(
            best_match_node, CacheTransferPhase.LOAD_BACK
        )[0]

        # Lock path & pre-evict if device pool is insufficient
        result = self.inc_lock_ref(best_match_node)
        ancestor_lock_params = result.to_dec_params()
        kv_tokens = len(kv_xfer.host_indices)

        # Build aux transfers, keyed per component.
        comp_xfers: dict[ComponentType, list] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            t = comp.build_hicache_transfers(
                best_match_node, CacheTransferPhase.LOAD_BACK, req=req
            )
            if t:
                comp_xfers[comp.component_type] = t
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.LOAD_BACK, kv_xfer, comp_xfers
        )

        # Skip if there is nothing to load, or if the Full-KV transfer is too
        # small / exceeds memory quota. Aux transfers should still run even
        # when the Full-KV load is skipped by thresholding.
        if (kv_tokens < self.load_back_threshold and not comp_xfers) or (
            mem_quota is not None and kv_tokens > mem_quota + result.delta
        ):
            self.dec_lock_ref(best_match_node, ancestor_lock_params)
            return False

        if self.supports_swa():
            avail = self.token_to_kv_pool_allocator.full_available_size()
        else:
            avail = self.token_to_kv_pool_allocator.available_size()
        if avail < kv_tokens:
            needed = kv_tokens - avail
            result = self.evict(EvictParams(num_tokens=needed))
            if result.num_tokens_evicted < needed:
                self.dec_lock_ref(best_match_node, ancestor_lock_params)
                return False

        # Load H→D
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        device_indices = self.cache_controller.load(
            host_indices=kv_xfer.host_indices,
            node_id=best_match_node.id,
            extra_pools=aux_xfers or None,
        )

        self.dec_lock_ref(best_match_node, ancestor_lock_params)
        if device_indices is None:
            return False

        # Commit: each component gets only its own transfers
        kv_xfer.device_indices = device_indices
        self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            best_match_node,
            CacheTransferPhase.LOAD_BACK,
            [kv_xfer],
        )
        for node in kv_xfer.nodes_to_load or ():
            self._record_store_event(node, medium=StorageMedium.GPU)
        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                best_match_node,
                CacheTransferPhase.LOAD_BACK,
                xfers,
            )

        self._update_evictable_leaf_sets(best_match_node)
        self.ongoing_load_back[best_match_node.id] = (
            best_match_node,
            self.inc_lock_ref(best_match_node).to_dec_params(),
        )
        if self.metrics_collector is not None and len(device_indices) > 0:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))
        if len(device_indices) > 0:
            logger.info(
                "init_load_back success: loaded %d tokens for node %d "
                "(mode=hicache_host restored_nodes=%d aux_transfers=%d) duration_ms=%.3f",
                len(device_indices),
                best_match_node.id,
                len(kv_xfer.nodes_to_load or ()),
                len(aux_xfers),
                (time.perf_counter() - start_time) * 1000.0,
            )
        return True

    def _build_sidecar_transfers(
        self,
        phase: CacheTransferPhase,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list[PoolTransfer]:
        transfers: list[PoolTransfer] = []
        for spec in self.sidecar_pool_specs:
            if spec.indices_from_pool == PoolName.KV:
                indices_source = kv_xfer
            else:
                source_component = {
                    PoolName.SWA: ComponentType.SWA,
                    PoolName.MAMBA: ComponentType.MAMBA,
                }.get(spec.indices_from_pool)
                if source_component is None:
                    raise AssertionError(
                        f"Unsupported sidecar indices source pool "
                        f"{spec.indices_from_pool}."
                    )
                matching_sources = comp_xfers.get(source_component, ())
                if not matching_sources:
                    continue
                indices_source = matching_sources[0]
                if indices_source.name != spec.indices_from_pool:
                    raise AssertionError(
                        f"Sidecar indices source pool {spec.indices_from_pool} "
                        f"resolved to {indices_source.name} during {phase}."
                    )

            indices = (
                indices_source.device_indices
                if phase == CacheTransferPhase.BACKUP_HOST
                else indices_source.host_indices
            )
            if indices is None or len(indices) == 0:
                continue
            transfers.append(
                PoolTransfer(
                    name=spec.pool_name,
                    hit_policy=spec.hit_policy,
                    indices_from_pool=spec.indices_from_pool,
                )
            )
        return transfers

    def _inc_hit_count(self, node: UnifiedTreeNode, chunked: bool = False) -> None:
        """Increment hit count; trigger write_backup when threshold reached."""
        if node.evicted or chunked:
            return
        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            return
        node.hit_count += 1
        if (
            self.cache_controller is not None
            and not node.backuped
            and node.hit_count >= self.write_through_threshold
        ):
            self.write_backup(node)

    def write_backup_storage(self, node: UnifiedTreeNode) -> None:
        if (
            not self.enable_storage
            or self.cache_controller is None
            or not node.backuped
        ):
            return
        if self._is_fluxon_hostless_full_mode():
            # Hostless Fluxon writes directly to external storage on the
            # device path. Marking storage_backed here without a matching
            # Fluxon put/ack cycle would create false-positive recoverability.
            return

        prefix_keys = None
        if self.hicache_storage_pass_prefix_keys:
            prefix_keys = node.get_prefix_hash_values(node.parent)

        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            transfers = comp.build_hicache_transfers(
                node,
                CacheTransferPhase.BACKUP_STORAGE,
            )
            if transfers:
                comp_xfers[comp.component_type] = transfers

        kv_xfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=node.component_data[BASE_COMPONENT_TYPE].host_value,
            keys=node.hash_value,
        )
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_STORAGE, kv_xfer, comp_xfers
        )
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)

        operation_id = self.cache_controller.write_storage(
            node.component_data[BASE_COMPONENT_TYPE].host_value,
            node.key.token_ids,
            node.hash_value,
            prefix_keys,
            extra_pools=aux_xfers or None,
        )
        logger.info(
            "HiCache storage backup submitted: node=%d tokens=%d aux_transfers=%d operation=%s",
            node.id,
            len(node.key),
            len(aux_xfers),
            operation_id,
        )
        self.ongoing_backup[operation_id] = (
            node,
            self.inc_host_lock_ref(node).to_dec_params(),
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: UnifiedTreeNode,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
        mamba_state_node: Optional[UnifiedTreeNode] = None,
    ) -> None:
        if not self.enable_storage or self.cache_controller is None:
            return
        if self._is_fluxon_hostless_full_mode():
            self._observe_fluxon_hostless_request(
                req_id,
                anchor_node_id=last_host_node.id,
                prefetch_input_tokens=len(new_input_tokens),
            )
            backend = self._fluxon_backend()
            if backend is None:
                self._observe_fluxon_hostless_request(
                    req_id,
                    prefetch_decision="backend_unavailable",
                )
                return

            mamba_state_key: str | None = None
            mamba_required_for_load_back = False
            if (
                self._is_fluxon_hostless_mamba_mode()
                and mamba_state_node is not None
                and self._hostless_mamba_needs_restore(mamba_state_node)
            ):
                mamba_state_key = self._node_mamba_storage_key(mamba_state_node)
                if mamba_state_key is None:
                    raise RuntimeError(
                        "Hostless Fluxon Mamba prefetch requires node hash: "
                        f"node={mamba_state_node.id}"
                    )
                mamba_required_for_load_back = True

            extra_key = last_host_node.key.extra_key if last_host_node.key else None
            prefetch_key = RadixKey(
                new_input_tokens,
                extra_key=extra_key,
                is_bigram=self.is_eagle,
            ).page_aligned(self.page_size)
            prefetch_length = len(prefetch_key)
            below_prefetch_threshold = prefetch_length < self.prefetch_threshold
            # The generic HiCache limiter models a materialized host pool.  The
            # Fluxon hostless path is metadata-only, so it must inspect the
            # source-aware Plan before charging holder and remote-source debt.
            kv_prefetch_enabled = not below_prefetch_threshold
            self._observe_fluxon_hostless_request(
                req_id,
                prefetch_input_tokens=prefetch_length,
                prefetch_decision=(
                    "below_threshold"
                    if below_prefetch_threshold
                    else "eligible"
                ),
            )

            hash_values: list[str] = []
            kv_handle = None
            transferable_pages = 0
            requested_pages = 0
            completed_tokens = 0
            has_ready_transfer = False
            atomic_group_lens: list[int] = []
            gpu_staging_lease = None
            kv_gpu_direct = False
            kv_handle_mode = "none"
            lineage_start_depth_pages = 0
            lineage_plan_unix_ns = 0
            lineage_plan_handle = -1
            kv_anchor_node: UnifiedTreeNode | None = None
            admission_active = False
            admission_total_tokens = 0
            admission_remote_pages = 0

            def cancel_kv_handle(caller: str) -> None:
                nonlocal kv_handle, gpu_staging_lease, kv_handle_mode
                self._cancel_fluxon_hostless_get_start_handle(
                    backend,
                    kv_handle,
                    caller,
                    gpu_direct=kv_gpu_direct,
                    plan_only=kv_handle_mode == "plan",
                )
                kv_handle = None
                kv_handle_mode = "none"
                if gpu_staging_lease is not None:
                    staging_lease = gpu_staging_lease
                    gpu_staging_lease = None
                    self._release_fluxon_gpu_staging_lease(
                        staging_lease,
                        caller,
                    )

            def release_prefetch_admission(caller: str) -> None:
                nonlocal admission_active
                if not admission_active:
                    return
                admission_active = False
                self._release_fluxon_hostless_prefetch_admission_values(
                    admission_total_tokens,
                    admission_remote_pages,
                    caller,
                )

            if kv_prefetch_enabled:
                kv_anchor_node = last_host_node
                while (
                    kv_anchor_node is not self.root_node
                    and kv_anchor_node.component_data[BASE_COMPONENT_TYPE].value is None
                ):
                    kv_anchor_node = kv_anchor_node.parent
                lineage_start_depth_pages = (
                    self._node_prefix_len(kv_anchor_node) // self.page_size
                )
                hash_values = self._node_hash_values_after_ancestor(
                    last_host_node,
                    kv_anchor_node,
                )
                group_cursor = last_host_node
                group_nodes = []
                while group_cursor is not kv_anchor_node and group_cursor is not self.root_node:
                    group_nodes.append(group_cursor)
                    group_cursor = group_cursor.parent
                if group_cursor is kv_anchor_node:
                    for group_node in reversed(group_nodes):
                        group_len = len(self._node_hash_values(group_node))
                        if group_len > 0:
                            atomic_group_lens.append(group_len)
                if not hash_values:
                    self._observe_fluxon_hostless_request(
                        req_id,
                        prefetch_decision="no_hash_values",
                    )
            requested_pages = len(hash_values)
            local_cpu_plan_pages = 0
            local_gpu_plan_pages = 0
            local_gpu_remote_indices: tuple[int, ...] = ()
            initial_start_at = time.perf_counter()
            if hash_values:
                try:
                    kv_handle = backend.get_plan(
                        hash_values,
                        prefix_best_effort=True,
                        atomic_group_lens=atomic_group_lens,
                    )
                    kv_handle_mode = "plan"
                    lineage_plan_unix_ns = time.time_ns()
                    lineage_plan_handle = int(
                        getattr(kv_handle, "backend_handle", -1)
                    )
                    local_cpu_plan_pages = int(
                        getattr(kv_handle.result, "transferable_len", 0)
                    )
                    local_gpu_plan_pages = int(
                        getattr(kv_handle.gpu_result, "transferable_len", 0)
                    )
                    local_gpu_remote_indices = tuple(
                        int(index)
                        for index in getattr(kv_handle, "gpu_remote_indices", ())
                    )
                except Exception as exc:
                    logger.warning(
                        "Fluxon hostless get_plan failed; treating as cache miss: "
                        "req=%s requested_pages=%d error=%s",
                        req_id,
                        requested_pages,
                        exc,
                    )
                    cancel_kv_handle("prefetch_get_plan_error")
                    self._observe_fluxon_hostless_request(
                        req_id,
                        prefetch_decision="get_plan_error",
                    )
            if kv_handle is not None:
                self._observe_fluxon_hostless_request(
                    req_id,
                    plan_ready_age_ms=self._fluxon_hostless_observation_age_ms(
                        req_id
                    ),
                )
            initial_start_ms = (time.perf_counter() - initial_start_at) * 1000.0

            plan_state_min = torch.tensor(
                [
                    local_cpu_plan_pages,
                    local_gpu_plan_pages,
                    int(mamba_required_for_load_back),
                ],
                dtype=torch.int,
            )
            plan_state_max = plan_state_min.clone()
            if self.tp_world_size > 1:
                torch.distributed.all_reduce(
                    plan_state_min,
                    op=torch.distributed.ReduceOp.MIN,
                    group=self.tp_group,
                )
                torch.distributed.all_reduce(
                    plan_state_max,
                    op=torch.distributed.ReduceOp.MAX,
                    group=self.tp_group,
                )
            cpu_common_pages = int(plan_state_min[0].item())
            gpu_common_pages = int(plan_state_min[1].item())
            self._observe_fluxon_hostless_request(
                req_id,
                requested_pages=requested_pages,
                initial_transferable_pages=local_cpu_plan_pages,
                gpu_plan_transferable_pages=local_gpu_plan_pages,
                tp_min_pages=cpu_common_pages,
                tp_max_pages=int(plan_state_max[0].item()),
                gpu_plan_tp_min_pages=gpu_common_pages,
                gpu_plan_tp_max_pages=int(plan_state_max[1].item()),
                initial_start_ms=initial_start_ms,
            )
            if int(plan_state_min[2].item()) != int(plan_state_max[2].item()):
                cancel_kv_handle("prefetch_tp_mamba_intent_mismatch")
                self._observe_fluxon_hostless_request(
                    req_id,
                    prefetch_decision="tp_mamba_intent_mismatch",
                )
                return
            if mamba_required_for_load_back and cpu_common_pages != requested_pages:
                cpu_common_pages = 0
                gpu_common_pages = 0

            gpu_common_remote_indices = tuple(
                index
                for index in local_gpu_remote_indices
                if index < gpu_common_pages
            )
            local_gpu_remote_pages = len(gpu_common_remote_indices)

            source_group_lens = atomic_group_lens or [1] * requested_pages
            for label, common_pages in (
                ("cpu", cpu_common_pages),
                ("gpu", gpu_common_pages),
            ):
                group_cursor_pages = 0
                for group_len in source_group_lens:
                    if group_cursor_pages + group_len > common_pages:
                        break
                    group_cursor_pages += group_len
                if group_cursor_pages != common_pages:
                    cancel_kv_handle(f"prefetch_{label}_common_prefix_split_atomic_group")
                    raise RuntimeError(
                        "Fluxon TP common plan prefix split an atomic group: "
                        f"req={req_id} mode={label} common={common_pages} "
                        f"group_pages={group_cursor_pages} groups={source_group_lens}"
                    )

            gpu_admission_block_reason = None
            if not _FLUXON_GPU_DIRECT_STAGING_ENABLED:
                gpu_admission_block_reason = "disabled"
            elif not kv_prefetch_enabled:
                gpu_admission_block_reason = "not_eligible"
            elif not hash_values:
                gpu_admission_block_reason = "no_hash_values"
            elif mamba_required_for_load_back:
                gpu_admission_block_reason = "mamba_required"
            elif gpu_common_pages <= 0:
                gpu_admission_block_reason = "no_gpu_transferable_prefix"
            elif gpu_common_pages != cpu_common_pages:
                gpu_admission_block_reason = "gpu_prefix_shorter_than_cpu"
            elif local_gpu_remote_pages <= 0:
                gpu_admission_block_reason = "no_remote_sources"
            (
                gpu_staging_lease,
                gpu_staging_admission,
            ) = backend.try_reserve_gpu_direct_staging(
                local_gpu_remote_pages,
                admission_block_reason=gpu_admission_block_reason,
            )
            reserve_attempt_age_ms = self._fluxon_hostless_observation_age_ms(
                req_id
            )
            self._observe_fluxon_hostless_request(
                req_id,
                gpu_reserve_attempt_age_ms=reserve_attempt_age_ms,
                gpu_reserve_age_ms=(
                    reserve_attempt_age_ms
                    if gpu_staging_lease is not None
                    else 0.0
                ),
            )
            local_gpu_reserved = int(gpu_staging_lease is not None)
            gpu_reservation_min = torch.tensor(
                [local_gpu_reserved, local_gpu_remote_pages],
                dtype=torch.int,
            )
            gpu_reservation_max = gpu_reservation_min.clone()
            if self.tp_world_size > 1:
                torch.distributed.all_reduce(
                    gpu_reservation_min,
                    op=torch.distributed.ReduceOp.MIN,
                    group=self.tp_group,
                )
                torch.distributed.all_reduce(
                    gpu_reservation_max,
                    op=torch.distributed.ReduceOp.MAX,
                    group=self.tp_group,
                )
            gpu_reservation_consistent = (
                int(gpu_reservation_min[0].item()) == 1
                and int(gpu_reservation_max[0].item()) == 1
            )
            gpu_reservation_tp_inconsistent = (
                int(gpu_reservation_min[0].item())
                != int(gpu_reservation_max[0].item())
            )
            if not gpu_reservation_consistent and gpu_staging_lease is not None:
                staging_lease = gpu_staging_lease
                gpu_staging_lease = None
                self._release_fluxon_gpu_staging_lease(
                    staging_lease,
                    "prefetch_gpu_reservation_fallback",
                )
            kv_gpu_direct = gpu_staging_lease is not None
            transferable_pages = (
                gpu_common_pages if kv_gpu_direct else cpu_common_pages
            )
            if kv_handle is not None and transferable_pages > 0:
                local_selected_remote_pages = sum(
                    1
                    for index in local_gpu_remote_indices
                    if index < transferable_pages
                )
                admission = self._try_acquire_fluxon_hostless_prefetch_admission(
                    req_id,
                    transferable_pages * self.page_size,
                    local_selected_remote_pages,
                )
                if bool(admission["admitted"]):
                    admission_total_tokens = int(admission["total_tokens"])
                    admission_remote_pages = int(admission["remote_pages"])
                    admission_active = True
                self._observe_fluxon_hostless_request(
                    req_id,
                    source_admission_reason=str(admission["reason"]),
                    source_admission_total_tokens=int(admission["total_tokens"]),
                    source_admission_remote_pages=int(admission["remote_pages"]),
                    source_admission_source_min_pages=int(
                        admission["source_min_pages"]
                    ),
                    source_admission_source_max_pages=int(
                        admission["source_max_pages"]
                    ),
                    source_admission_total_before=int(admission["total_before"]),
                    source_admission_total_after=int(admission["total_after"]),
                    source_admission_remote_before=int(admission["remote_before"]),
                    source_admission_remote_after=int(admission["remote_after"]),
                    source_admission_device_reclaimable_min=int(
                        admission["device_reclaimable_min"]
                    ),
                    source_admission_device_reclaimable_max=int(
                        admission["device_reclaimable_max"]
                    ),
                    source_admission_device_prefetch_budget=int(
                        admission["device_prefetch_budget"]
                    ),
                )
                if not admission_active:
                    reason = str(admission["reason"])
                    cancel_kv_handle(f"prefetch_source_admission_{reason}")
                    self._observe_fluxon_hostless_request(
                        req_id,
                        prefetch_decision=f"source_admission_{reason}",
                        final_transferable_pages=0,
                    )
                    return
            selected_gpu_remote_indices: tuple[int, ...] = ()
            planned_execute_pages = transferable_pages
            planned_execute_gpu_direct = kv_gpu_direct
            local_plan_execute_error: Exception | None = None
            tp_plan_execute_commit_rejected = False
            if kv_handle is not None and transferable_pages > 0:
                try:
                    if kv_gpu_direct:
                        kv_handle = backend.execute_get_plan_gpu(
                            kv_handle,
                            gpu_staging_lease,
                            consume_prefix_len=transferable_pages,
                        )
                        selected_gpu_remote_indices = tuple(
                            int(index) for index in kv_handle.remote_indices
                        )
                        kv_handle_mode = "gpu"
                        self._observe_fluxon_hostless_request(
                            req_id,
                            gpu_execute_return_age_ms=(
                                self._fluxon_hostless_observation_age_ms(req_id)
                            ),
                            gpu_backend_handle=int(
                                getattr(kv_handle, "backend_handle", -1)
                            ),
                        )
                    else:
                        kv_handle = backend.execute_get_plan_cpu(
                            kv_handle,
                            consume_prefix_len=transferable_pages,
                        )
                        kv_handle_mode = "cpu"
                except Exception as exc:
                    logger.warning(
                        "Fluxon hostless plan execute failed; treating as cache miss: "
                        "req=%s mode=%s pages=%d error=%s",
                        req_id,
                        "gpu" if kv_gpu_direct else "cpu",
                        transferable_pages,
                        exc,
                    )
                    cancel_kv_handle("prefetch_plan_execute_error")
                    release_prefetch_admission("prefetch_plan_execute_error")
                    transferable_pages = 0
                    if gpu_staging_lease is not None:
                        staging_lease = gpu_staging_lease
                        gpu_staging_lease = None
                        self._release_fluxon_gpu_staging_lease(
                            staging_lease,
                            "prefetch_plan_execute_error",
                        )
                    kv_gpu_direct = False
                    self._observe_fluxon_hostless_request(
                        req_id,
                        prefetch_decision="plan_execute_error",
                    )
                    local_plan_execute_error = exc
            elif kv_handle is not None:
                cancel_kv_handle("prefetch_zero_common_plan")

            if planned_execute_pages > 0 and self.tp_world_size > 1:
                expected_mode = "gpu" if planned_execute_gpu_direct else "cpu"
                local_plan_execute_succeeded = bool(
                    kv_handle is not None
                    and kv_handle_mode == expected_mode
                    and transferable_pages == planned_execute_pages
                )
                (
                    tp_plan_execute_committed,
                    execute_succeeded_ranks,
                    execute_gpu_direct_ranks,
                ) = self._fluxon_hostless_tp_plan_execute_commit(
                    local_succeeded=local_plan_execute_succeeded,
                    gpu_direct=planned_execute_gpu_direct,
                )
                if not tp_plan_execute_committed:
                    tp_plan_execute_commit_rejected = True
                    logger.warning(
                        "Fluxon hostless TP plan execute commit rejected: "
                        "req=%s mode=%s pages=%d local_succeeded=%s "
                        "succeeded_ranks=%d gpu_direct_ranks=%d "
                        "tp_world_size=%d local_error=%s",
                        req_id,
                        expected_mode,
                        planned_execute_pages,
                        local_plan_execute_succeeded,
                        execute_succeeded_ranks,
                        execute_gpu_direct_ranks,
                        self.tp_world_size,
                        local_plan_execute_error,
                    )
                    cancel_kv_handle("prefetch_tp_plan_execute_commit_rejected")
                    release_prefetch_admission(
                        "prefetch_tp_plan_execute_commit_rejected"
                    )
                    transferable_pages = 0
                    kv_gpu_direct = False
                    selected_gpu_remote_indices = ()
                    self._observe_fluxon_hostless_request(
                        req_id,
                        prefetch_decision="tp_plan_execute_commit_rejected",
                    )

            if (
                transferable_pages > 0
                and kv_handle is not None
                and kv_handle_mode in ("cpu", "gpu")
            ):
                lineage_keys = hash_values[:transferable_pages]
                remote_indices = set(local_gpu_remote_indices)
                lineage_sources = "".join(
                    "R"
                    if index in remote_indices
                    else "L"
                    if index < local_gpu_plan_pages
                    else "U"
                    for index in range(transferable_pages)
                )
                self._observe_fluxon_hostless_request(
                    req_id,
                    lineage_plan_unix_ns=lineage_plan_unix_ns,
                    lineage_plan_handle=lineage_plan_handle,
                    lineage_start_depth_pages=lineage_start_depth_pages,
                    lineage_cpu_plan_pages=local_cpu_plan_pages,
                    lineage_gpu_plan_pages=local_gpu_plan_pages,
                    lineage_materialization=(
                        "gdr_h2d" if kv_handle_mode == "gpu" else "cpu_h2d"
                    ),
                    lineage_key_ids=tuple(
                        self._fluxon_kv_lineage_key_id(key)
                        for key in lineage_keys
                    ),
                    lineage_sources=lineage_sources,
                )

            gpu_direct_admission_reason = str(gpu_staging_admission["reason"])
            if tp_plan_execute_commit_rejected:
                gpu_direct_admission_reason = "tp_plan_execute_commit_rejected"
            elif gpu_reservation_tp_inconsistent:
                gpu_direct_admission_reason = "tp_reservation_inconsistent"
            self._observe_fluxon_hostless_request(
                req_id,
                final_transferable_pages=transferable_pages,
                gpu_direct_local_reserved=local_gpu_reserved,
                gpu_direct_selected=int(kv_gpu_direct),
                gpu_direct_admission_reason=gpu_direct_admission_reason,
                gpu_direct_requested_pages=int(
                    gpu_staging_admission["requested_pages"]
                ),
                gpu_direct_capacity_slots=int(
                    gpu_staging_admission["capacity_slots"]
                ),
                gpu_direct_free_slots_before=int(
                    gpu_staging_admission["free_slots_before"]
                ),
                gpu_direct_live_slots_before=int(
                    gpu_staging_admission["live_slots_before"]
                ),
                gpu_direct_active_leases_before=int(
                    gpu_staging_admission["active_leases_before"]
                ),
                gpu_direct_free_slots_after=int(
                    gpu_staging_admission["free_slots_after"]
                ),
                gpu_direct_live_slots_after=int(
                    gpu_staging_admission["live_slots_after"]
                ),
                gpu_direct_active_leases_after=int(
                    gpu_staging_admission["active_leases_after"]
                ),
                gpu_direct_high_watermark_slots=int(
                    gpu_staging_admission["high_watermark_slots"]
                ),
                gpu_direct_tp_min_pages=int(gpu_reservation_min[1].item()),
                gpu_direct_tp_max_pages=int(gpu_reservation_max[1].item()),
            )
            if (
                self.tp_world_size == 1
                and not hash_values
                and not mamba_required_for_load_back
            ):
                return
            if not hash_values and not mamba_required_for_load_back:
                observation = self._observe_fluxon_hostless_request(req_id)
                if observation.get("prefetch_decision") == "eligible":
                    observation["prefetch_decision"] = "no_hash_values"
                return
            if hash_values:
                if mamba_required_for_load_back and transferable_pages != requested_pages:
                    cancel_kv_handle("prefetch_mamba_requires_full_kv_prefix")
                    hash_values = []
                    transferable_pages = 0
                else:
                    hash_values = hash_values[:transferable_pages]
                    completed_tokens = transferable_pages * self.page_size
                    has_ready_transfer = transferable_pages > 0
                    if transferable_pages == 0:
                        cancel_kv_handle("prefetch_zero_transferable_kv")
                        observation = self._observe_fluxon_hostless_request(req_id)
                        if observation.get("prefetch_decision") in (
                            "eligible",
                            "tp_common_prefix_reused",
                        ):
                            observation["prefetch_decision"] = "zero_transferable"
                        observation["final_transferable_pages"] = 0
            mamba_handle = None
            if (
                self._is_fluxon_hostless_mamba_mode()
                and mamba_required_for_load_back
                and (requested_pages == 0 or completed_tokens > 0)
            ):
                local_mamba_hit = 0
                try:
                    if mamba_state_key is None:
                        raise RuntimeError(
                            "Hostless Fluxon Mamba prefetch requires a state key"
                        )
                    mamba_handle = backend.get_start(
                        [mamba_state_key],
                        component_name=PoolName.MAMBA,
                        prefix_best_effort=False,
                        atomic_group_lens=[1],
                    )
                    mamba_result = mamba_handle.result
                    local_mamba_hit = int(
                        int(getattr(mamba_result, "transferable_len", 0)) == 1
                        and bool(getattr(mamba_result, "all_hit", False))
                    )
                except Exception as exc:
                    logger.warning(
                        "Fluxon hostless Mamba get_start failed; treating as cache miss: "
                        "req=%s key=%s error=%s",
                        req_id,
                        mamba_state_key,
                        exc,
                    )
                    self._cancel_fluxon_hostless_get_start_handle(
                        backend,
                        mamba_handle,
                        "prefetch_mamba_get_start_error",
                    )
                    mamba_handle = None
                    local_mamba_hit = 0
                if self.tp_world_size > 1:
                    mamba_hit_tensor = torch.tensor(
                        local_mamba_hit,
                        dtype=torch.int,
                    )
                    torch.distributed.all_reduce(
                        mamba_hit_tensor,
                        op=torch.distributed.ReduceOp.MIN,
                        group=self.tp_group,
                    )
                    local_mamba_hit = int(mamba_hit_tensor.item())
                if local_mamba_hit != 1:
                    completed_tokens = 0
                    has_ready_transfer = False
                    cancel_kv_handle("prefetch_mamba_zero_transferable")
                    self._cancel_fluxon_hostless_get_start_handle(
                        backend,
                        mamba_handle,
                        "prefetch_mamba_zero_transferable",
                    )
                    mamba_handle = None
                elif not hash_values and mamba_required_for_load_back:
                    has_ready_transfer = True
            if not has_ready_transfer:
                cancel_kv_handle("prefetch_no_ready_transfer")
                release_prefetch_admission("prefetch_no_ready_transfer")
                self._cancel_fluxon_hostless_get_start_handle(
                    backend,
                    mamba_handle,
                    "prefetch_no_ready_transfer",
                )
                observation = self._observe_fluxon_hostless_request(req_id)
                if observation.get("prefetch_decision") in ("eligible", "entered"):
                    observation["prefetch_decision"] = "no_ready_transfer"
                return
            total_tokens = (
                len(hash_values) * self.page_size
                if hash_values
                else (1 if mamba_handle is not None else 0)
            )
            if admission_active and total_tokens != admission_total_tokens:
                release_prefetch_admission("prefetch_admission_size_mismatch")
                cancel_kv_handle("prefetch_admission_size_mismatch")
                raise RuntimeError(
                    "Fluxon hostless admission size changed after Plan execution: "
                    f"req={req_id} admitted={admission_total_tokens} "
                    f"operation={total_tokens}"
                )
            try:
                operation = _FluxonHostlessPrefetchOperation(
                    backend=backend,
                    hash_value=hash_values,
                    kv_handle=kv_handle,
                    mamba_handle=mamba_handle,
                    mamba_key=mamba_state_key if mamba_handle is not None else None,
                    completed_tokens=completed_tokens,
                    total_tokens=total_tokens,
                    anchor_node_id=last_host_node.id,
                    mamba_anchor_node_id=(
                        mamba_state_node.id
                        if mamba_state_node is not None
                        else last_host_node.id
                    ),
                    has_ready_transfer=has_ready_transfer,
                    gpu_staging_lease=gpu_staging_lease,
                    gpu_remote_indices=selected_gpu_remote_indices,
                    admission_total_tokens=admission_total_tokens,
                    admission_remote_pages=admission_remote_pages,
                    admission_active=admission_active,
                )
            except Exception:
                release_prefetch_admission("prefetch_operation_create_error")
                raise
            admission_active = False
            if hash_values:
                if kv_anchor_node is None:
                    self._cancel_fluxon_hostless_prefetch_operation(
                        operation,
                        "prefetch_missing_device_anchor",
                    )
                    raise RuntimeError(
                        "Fluxon hostless KV prefetch lost its device anchor"
                    )
                try:
                    self._acquire_fluxon_hostless_anchor_lock(
                        operation,
                        kv_anchor_node,
                    )
                except Exception:
                    self._cancel_fluxon_hostless_prefetch_operation(
                        operation,
                        "prefetch_device_anchor_lock_error",
                    )
                    raise
            gpu_staging_lease = None
            self.ongoing_prefetch[req_id] = (
                last_host_node,
                prefetch_key,
                None,
                operation,
                None,
                {},
            )
            previous_decision = self._observe_fluxon_hostless_request(req_id).get(
                "prefetch_decision"
            )
            self._observe_fluxon_hostless_request(
                req_id,
                prefetch_decision=(
                    "submitted_after_tp_prefix_reuse"
                    if previous_decision == "tp_common_prefix_reused"
                    else "submitted"
                ),
                requested_pages=requested_pages,
                final_transferable_pages=transferable_pages,
                device_anchor_node_id=(
                    int(kv_anchor_node.id) if kv_anchor_node is not None else -1
                ),
            )
            logger.info(
                "HiCache prefetch submitted: req=%s tokens=%d "
                "mode=fluxon_hostless_full pages=%d transferable=%d "
                "gpu_direct=%s mamba=%s device_anchor=%d occupied=%d",
                req_id,
                total_tokens,
                requested_pages,
                transferable_pages,
                operation.gpu_direct,
                mamba_handle is not None,
                int(kv_anchor_node.id) if kv_anchor_node is not None else -1,
                self.cache_controller.prefetch_tokens_occupied,
            )
            return

        extra_key = last_host_node.key.extra_key if last_host_node.key else None
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        if (
            prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            return

        anchor_lock_params = self.inc_host_lock_ref(last_host_node).to_dec_params()
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            self.evict_host(prefetch_length)
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            available_size = self.cache_controller.mem_pool_host.available_size()
            prefetch_length = available_size - (available_size % self.page_size)
            if prefetch_length >= self.prefetch_threshold:
                prefetch_key = prefetch_key[:prefetch_length]
                host_indices = self.cache_controller.mem_pool_host.alloc(
                    prefetch_length
                )
            else:
                self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                return
        if host_indices is None:
            self.dec_host_lock_ref(last_host_node, anchor_lock_params)
            return

        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        alloc_failed = False
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            transfers = comp.build_hicache_transfers(
                last_host_node,
                CacheTransferPhase.PREFETCH,
                token_ids=prefetch_key.token_ids,
                prefetch_tokens=len(prefetch_key),
                last_hash=last_hash,
            )
            if transfers == []:
                alloc_failed = True
                break
            if transfers:
                comp_xfers[comp.component_type] = transfers
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.PREFETCH, kv_xfer, comp_xfers
        )
        if alloc_failed:
            self.cache_controller.append_host_mem_release(
                host_indices=host_indices,
                extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
            )
            self.dec_host_lock_ref(last_host_node, anchor_lock_params)
            return

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            prefetch_key.token_ids,
            last_hash,
            prefix_keys,
            extra_pools=aux_xfers or None,
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        )
        self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)
        logger.info(
            "HiCache prefetch submitted: req=%s tokens=%d mode=hicache_host aux_transfers=%d occupied=%d",
            req_id,
            len(prefetch_key),
            len(aux_xfers),
            self.cache_controller.prefetch_tokens_occupied,
        )

    def _prefetch_timeout_check_linear_func(self, operation) -> bool:
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation) -> bool:
        if self.prefetch_stop_policy == "best_effort":
            return True

        if isinstance(operation, _FluxonHostlessPrefetchOperation):
            completed = operation.is_finished()
        elif len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self._prefetch_timeout_check_linear_func(
                operation
            )
        else:
            return True

        operation_terminated = operation.is_terminated()
        states = torch.tensor(
            [1 - int(can_terminate), int(operation_terminated)],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                states, op=torch.distributed.ReduceOp.MAX, group=self.tp_group
            )
        can_terminate = states[0].item() == 0
        operation_terminated = states[1].item() == 1
        return can_terminate or operation_terminated

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            return True

        (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[req_id]
        if isinstance(operation, _FluxonHostlessPrefetchOperation):
            if not self.can_terminate_prefetch(operation):
                return False
            local_completed_tokens = operation.completed_tokens
            min_completed_tokens = local_completed_tokens
            max_completed_tokens = local_completed_tokens
            if self.tp_world_size > 1:
                completed_tokens_tensor = torch.tensor(
                    min_completed_tokens, dtype=torch.int
                )
                max_completed_tokens_tensor = completed_tokens_tensor.clone()
                torch.distributed.all_reduce(
                    completed_tokens_tensor,
                    op=torch.distributed.ReduceOp.MIN,
                    group=self.tp_group,
                )
                torch.distributed.all_reduce(
                    max_completed_tokens_tensor,
                    op=torch.distributed.ReduceOp.MAX,
                    group=self.tp_group,
                )
                min_completed_tokens = int(completed_tokens_tensor.item())
                max_completed_tokens = int(max_completed_tokens_tensor.item())
            if min_completed_tokens != max_completed_tokens:
                operation.completed_tokens = 0
                operation.has_ready_transfer = False
                min_completed_tokens = 0
                self._observe_fluxon_hostless_request(
                    req_id,
                    prefetch_decision="completed_tokens_rank_mismatch",
                )
            else:
                operation.completed_tokens = min_completed_tokens
            if operation.hash_value:
                if min_completed_tokens % self.page_size != 0:
                    raise RuntimeError(
                        "Fluxon hostless prefetch completed tokens must be page aligned: "
                        f"completed={min_completed_tokens} page_size={self.page_size}"
                    )
            local_prepare_error: Exception | None = None
            get_transfer_started_at = time.perf_counter()
            if operation.has_ready_transfer:
                try:
                    if operation.kv_handle is not None:
                        transfer_consume_start_age_ms = (
                            self._fluxon_hostless_observation_age_ms(req_id)
                        )
                        self._observe_fluxon_hostless_request(
                            req_id,
                            transfer_consume_start_age_ms=(
                                transfer_consume_start_age_ms
                            ),
                        )
                        if operation.gpu_direct:
                            gpu_handle = operation.kv_handle
                            operation.kv_plan_ptr = operation.backend.get_transfer_gpu(
                                gpu_handle,
                                consume_prefix_len=len(operation.hash_value),
                            )
                            transfer_wall_ms = float(
                                gpu_handle.transfer_wall_us
                            ) / 1000.0
                            finish_wait_ms = float(
                                gpu_handle.finish_wait_us
                            ) / 1000.0
                            terminal_to_consume_ms = float(
                                gpu_handle.terminal_to_consume_us
                            ) / 1000.0
                            terminal_before_consume = bool(
                                gpu_handle.terminal_before_consume
                            )
                            terminal_age_ms = (
                                transfer_consume_start_age_ms
                                - terminal_to_consume_ms
                                if terminal_before_consume
                                else transfer_consume_start_age_ms
                                + finish_wait_ms
                            )
                            rdma_start_age_ms = max(
                                0.0,
                                terminal_age_ms - transfer_wall_ms,
                            )
                            self._observe_fluxon_hostless_request(
                                req_id,
                                gpu_backend_handle=int(
                                    getattr(gpu_handle, "backend_handle", -1)
                                ),
                                rdma_start_age_ms=rdma_start_age_ms,
                                rdma_terminal_age_ms=max(0.0, terminal_age_ms),
                                rdma_transfer_wall_ms=transfer_wall_ms,
                                rdma_terminal_before_consume=int(
                                    terminal_before_consume
                                ),
                                rdma_terminal_to_consume_ms=(
                                    terminal_to_consume_ms
                                ),
                                rdma_finish_wait_ms=finish_wait_ms,
                            )
                            operation.kv_handle = None
                            if operation.gpu_staging_lease is None:
                                raise RuntimeError(
                                    "Fluxon GPU transfer lost its staging lease"
                                )
                            operation.gpu_staging_lease.trim_after_transfer(
                                len(operation.gpu_remote_indices)
                            )
                        else:
                            operation.kv_plan_ptr = operation.backend.get_transfer(
                                operation.kv_handle,
                                consume_prefix_len=len(operation.hash_value),
                            )
                            operation.kv_handle = None
                    if operation.mamba_handle is not None:
                        operation.mamba_plan_ptr = operation.backend.get_transfer(
                            operation.mamba_handle
                        )
                        operation.mamba_handle = None
                except Exception as exc:
                    local_prepare_error = exc
                    logger.warning(
                        "Fluxon hostless prepare finish failed: req=%s "
                        "completed_local=%d completed_synced=%d error=%s",
                        req_id,
                        local_completed_tokens,
                        min_completed_tokens,
                        exc,
                    )
                    self._cancel_fluxon_hostless_prefetch_operation(
                        operation,
                        "prefetch_prepare_finish_error",
                    )
                    min_completed_tokens = 0
                    operation.completed_tokens = 0
                    operation.has_ready_transfer = False
                    self._observe_fluxon_hostless_request(
                        req_id,
                        prefetch_decision="get_transfer_error",
                    )
            get_transfer_ms = (
                time.perf_counter() - get_transfer_started_at
            ) * 1000.0
            local_prepare_succeeded = bool(operation.has_ready_transfer)
            if operation.hash_value and operation.kv_plan_ptr is None:
                local_prepare_succeeded = False
            if operation.mamba_key is not None and operation.mamba_plan_ptr is None:
                local_prepare_succeeded = False

            # A prepared restore is visible to the radix tree only after every
            # TP rank owns a usable transfer plan.  A local get_transfer error
            # must therefore turn into one shared cache miss; otherwise one
            # rank restores the prefix while its peer recomputes it.
            prepared_rank_count = int(local_prepare_succeeded)
            if self.tp_world_size > 1:
                prepared_rank_count_tensor = torch.tensor(
                    prepared_rank_count,
                    dtype=torch.int,
                )
                torch.distributed.all_reduce(
                    prepared_rank_count_tensor,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.tp_group,
                )
                prepared_rank_count = int(prepared_rank_count_tensor.item())
            tp_prepare_succeeded = prepared_rank_count == self.tp_world_size
            if not tp_prepare_succeeded:
                logger.warning(
                    "Fluxon hostless TP prepare commit rejected: req=%s "
                    "local_succeeded=%s prepared_ranks=%d tp_world_size=%d "
                    "completed_local=%d completed_synced=%d local_error=%s",
                    req_id,
                    local_prepare_succeeded,
                    prepared_rank_count,
                    self.tp_world_size,
                    local_completed_tokens,
                    min_completed_tokens,
                    local_prepare_error,
                )
                self._cancel_fluxon_hostless_prefetch_operation(
                    operation,
                    "prefetch_tp_prepare_commit_rejected",
                )
                min_completed_tokens = 0
                operation.completed_tokens = 0
                operation.has_ready_transfer = False
                observation = self._observe_fluxon_hostless_request(req_id)
                if observation.get("prefetch_decision") in (
                    "submitted",
                    "submitted_after_tp_prefix_reuse",
                ):
                    observation[
                        "prefetch_decision"
                    ] = "tp_prepare_commit_rejected"
            del self.ongoing_prefetch[req_id]
            self._release_fluxon_hostless_prefetch_admission(
                operation,
                "prefetch_ready",
            )
            if operation.has_ready_transfer:
                self.fluxon_hostless_ready_prefetch[req_id] = operation
                observation = self._observe_fluxon_hostless_request(req_id)
                ready_pages = min_completed_tokens // self.page_size
                self._observe_fluxon_hostless_request(
                    req_id,
                    ready_pages=ready_pages,
                    ready_wait_ms=(
                        time.monotonic() - observation["created_at"]
                    )
                    * 1000.0,
                    get_transfer_ms=get_transfer_ms,
                    gpu_direct=int(operation.gpu_direct),
                )
            else:
                self._cancel_fluxon_hostless_prefetch_operation(
                    operation,
                    "prefetch_complete_zero_hit",
                )
                observation = self._observe_fluxon_hostless_request(req_id)
                if observation.get("prefetch_decision") in (
                    "submitted",
                    "submitted_after_tp_prefix_reuse",
                ):
                    observation["prefetch_decision"] = "prefetch_complete_zero_hit"
                observation["get_transfer_ms"] = get_transfer_ms
            self.prefetch_loaded_tokens_by_reqid[req_id] = min_completed_tokens
            logger.info(
                "HiCache prefetch success req=%s mode=fluxon_hostless_full "
                "gpu_direct=%s completed_local=%d completed_synced=%d occupied=%d",
                req_id,
                operation.gpu_direct,
                local_completed_tokens,
                min_completed_tokens,
                self.cache_controller.prefetch_tokens_occupied,
            )
            if (
                self.enable_storage_metrics
                and self.storage_metrics_collector is not None
                and min_completed_tokens > 0
            ):
                self.storage_metrics_collector.log_prefetched_tokens(
                    min_completed_tokens
                )
            return True
        if operation.host_indices is None:
            return True
        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        min_completed_tokens = completed_tokens
        if self.tp_world_size > 1:
            completed_tokens_tensor = torch.tensor(
                min_completed_tokens, dtype=torch.int
            )
            torch.distributed.all_reduce(
                completed_tokens_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            min_completed_tokens = int(completed_tokens_tensor.item())

        fetched_key = prefetch_key[:min_completed_tokens]
        insert_result = self._insert_helper_host(
            last_host_node,
            fetched_key,
            host_indices[:min_completed_tokens],
            hash_value[: min_completed_tokens // self.page_size],
        )

        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                last_host_node,
                CacheTransferPhase.PREFETCH,
                xfers,
                insert_result=insert_result,
                pool_storage_result=operation.pool_storage_result,
            )

        self.cache_controller.mem_pool_host.free(
            host_indices[: insert_result.prefix_len]
        )
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        self.dec_host_lock_ref(last_host_node, anchor_lock_params)
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        loaded_from_storage = min_completed_tokens - insert_result.prefix_len
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage
        logger.info(
            "HiCache prefetch success req=%s completed_local=%d completed_synced=%d matched=%d loaded=%d tail_release=%d occupied=%d",
            req_id,
            completed_tokens,
            min_completed_tokens,
            insert_result.prefix_len,
            loaded_from_storage,
            completed_tokens - min_completed_tokens,
            self.cache_controller.prefetch_tokens_occupied,
        )
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)
        return True

    def terminate_prefetch(self, req_id: str) -> None:
        if req_id not in self.ongoing_prefetch:
            return
        _, _, _, operation, _, _ = self.ongoing_prefetch[req_id]
        if isinstance(operation, _FluxonHostlessPrefetchOperation):
            operation.mark_terminate()
            return
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def release_aborted_request(self, rid: str) -> None:
        had_fluxon_observation = rid in self._fluxon_hostless_request_observations
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        ready_operation = self.fluxon_hostless_ready_prefetch.pop(rid, None)
        if ready_operation is not None:
            self._cancel_fluxon_hostless_prefetch_operation(
                ready_operation,
                "release_aborted_ready_prefetch",
            )
        if rid not in self.ongoing_prefetch:
            if had_fluxon_observation:
                self._finish_fluxon_hostless_request_observation(
                    rid,
                    (
                        "request_aborted_ready"
                        if ready_operation is not None
                        else "request_aborted_no_prefetch"
                    ),
                )
            return

        (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[rid]
        if isinstance(operation, _FluxonHostlessPrefetchOperation):
            self._cancel_fluxon_hostless_prefetch_operation(
                operation,
                "release_aborted_ongoing_prefetch",
            )
            del self.ongoing_prefetch[rid]
            if had_fluxon_observation:
                self._finish_fluxon_hostless_request_observation(
                    rid,
                    "request_aborted_ongoing_prefetch",
                )
            return
        if operation.host_indices is None:
            if had_fluxon_observation:
                self._finish_fluxon_hostless_request_observation(
                    rid,
                    "request_aborted_empty_operation",
                )
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        if self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)
        self.dec_host_lock_ref(last_host_node, anchor_lock_params)
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(
            host_indices=host_indices[:completed_tokens],
            extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
        )
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)
        if had_fluxon_observation:
            self._finish_fluxon_hostless_request_observation(
                rid,
                "request_aborted_host_prefetch",
            )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        extra_release_counts: Optional[dict[PoolName, int]],
        log_metrics: bool,
    ) -> None:
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            drained = 0
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is None:
                    continue
                drained += 1
                (
                    last_host_node,
                    prefetch_key,
                    _host_indices,
                    operation,
                    anchor_lock_params,
                    comp_xfers,
                ) = info
                if isinstance(operation, _FluxonHostlessPrefetchOperation):
                    self._cancel_fluxon_hostless_prefetch_operation(
                        operation,
                        "prefetch_revoke",
                    )
                    if req_id in self._fluxon_hostless_request_observations:
                        self._finish_fluxon_hostless_request_observation(
                            req_id,
                            "prefetch_revoked",
                        )
                    continue
                cc.append_host_mem_release(
                    extra_pools=[x for xfers in comp_xfers.values() for x in xfers]
                )
                self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                cc.prefetch_tokens_occupied -= len(prefetch_key)
                if cc.prefetch_tokens_occupied < 0:
                    cc.prefetch_tokens_occupied = 0
            return drained

        def _drain_backup():
            drained = 0
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                drained += 1
                entry = self.ongoing_backup.pop(operation.id, None)
                if entry is not None:
                    node, lock_params = entry
                    self.dec_host_lock_ref(node, lock_params)
                if (
                    log_metrics
                    and self.enable_storage_metrics
                    and self.storage_metrics_collector is not None
                ):
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )
            return drained

        def _drain_release():
            host_indices_list = []
            released_tokens = 0
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
                released_tokens += len(host_indices)
            if host_indices_list:
                cc.mem_pool_host.free(torch.cat(host_indices_list, dim=0))
            return len(host_indices_list), released_tokens

        def _drain_extra_release():
            drained: dict[PoolName, tuple[int, int]] = {}
            if not extra_release_counts:
                return drained
            for pool_name, limit in extra_release_counts.items():
                release_queue = cc.extra_host_mem_release_queues.get(pool_name)
                if release_queue is None:
                    continue
                host_indices_list = []
                released_tokens = 0
                for host_indices in _drain_queue(release_queue, limit):
                    host_indices_list.append(host_indices)
                    released_tokens += len(host_indices)
                if host_indices_list:
                    entry = cc.mem_pool_host.entry_map.get(pool_name)
                    if entry is not None:
                        entry.host_pool.free(torch.cat(host_indices_list, dim=0))
                drained[pool_name] = (len(host_indices_list), released_tokens)
            return drained

        _drain_revoke()
        _drain_backup()
        _drain_release()
        _drain_extra_release()

    def drain_storage_control_queues(self) -> None:
        cc = self.cache_controller
        extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
        extra_pool_names = list(extra_release_queues)
        local_qsize_list = [
            cc.prefetch_revoke_queue.qsize(),
            cc.ack_backup_queue.qsize(),
            cc.host_mem_release_queue.qsize(),
            *[
                extra_release_queues[pool_name].qsize()
                for pool_name in extra_pool_names
            ],
        ]
        qsizes = torch.tensor(
            local_qsize_list,
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        qsize_list = list(map(int, qsizes.tolist()))
        n_revoke, n_backup, n_release = qsize_list[:3]
        extra_release_counts = {
            pool_name: count
            for pool_name, count in zip(extra_pool_names, qsize_list[3:])
        }
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            extra_release_counts=extra_release_counts,
            log_metrics=True,
        )

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[dict[str, str]],
    ) -> None:
        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            attn_cp_rank, attn_cp_size = (
                self.cache_controller.get_attn_cp_rank_and_size()
            )
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
                "attn_cp_rank": attn_cp_rank,
                "attn_cp_size": attn_cp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            existing_collector = self.storage_metrics_collector
            if existing_collector is None:
                self.storage_metrics_collector = StorageMetricsCollector(labels=labels)
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                existing_collector.labels = labels
            else:
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )
        else:
            self.storage_metrics_collector = None

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        return (
            False,
            "UnifiedRadixCache does not support runtime HiCache storage attach yet. "
            "Configure hicache_storage_backend at startup instead.",
        )

    def detach_storage_backend(self) -> tuple[bool, str]:
        return (
            False,
            "UnifiedRadixCache does not support runtime HiCache storage detach yet. "
            "Restart without hicache_storage_backend to disable it.",
        )

    def clear_storage_backend(self) -> bool:
        try:
            ok = self.cache_controller.clear_storage_backend()
        except Exception as e:
            logger.error("Failed to clear hierarchical cache storage backend: %s", e)
            return False
        if ok:
            logger.info("Hierarchical cache storage backend cleared successfully!")
        return ok

    # ---- HiCache: Async Event Management ----

    def writing_check(self, write_back: bool = False) -> None:
        """Poll write-through completions."""
        cc = self.cache_controller
        if cc is None:
            return
        self._drain_fluxon_hostless_acks(block=False)

        if write_back:
            # Blocking: wait for all pending write-backs
            while self.ongoing_write_through:
                for _, finish_event, ack_list in cc.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        entry = self.ongoing_write_through.pop(ack_id, None)
                        if entry is not None:
                            node, params = entry
                            self._record_store_event(node, medium=StorageMedium.CPU)
                            if params is not None:
                                self.dec_lock_ref(node, params)
                            if self.enable_storage:
                                self.write_backup_storage(node)
                cc.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            while True:
                progressed = False
                for node_id in list(self.ongoing_fluxon_hostless_backup):
                    batch = self.ongoing_fluxon_hostless_backup.get(node_id)
                    if batch is None or not batch.write_back:
                        continue
                    self._finish_fluxon_hostless_write_batch(
                        node_id,
                        block=True,
                        caller="writing_check(write_back=True)",
                    )
                    progressed = True
                pending_write_back = any(
                    batch.write_back for batch in self.ongoing_fluxon_hostless_backup.values()
                )
                if not pending_write_back:
                    break
                if not progressed:
                    break
            self._drain_fluxon_hostless_acks(block=True)
            return

        if len(self.ongoing_write_through) == 0:
            for node_id in list(self.ongoing_fluxon_hostless_backup):
                self._finish_fluxon_hostless_write_batch(
                    node_id,
                    block=False,
                    caller="writing_check(no_write_through)",
                )
            return

        finish_count = 0
        for _, finish_event, ack_list in cc.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1

        # TP sync: MIN across all ranks for consistent tree updates
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_size, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        finish_count = int(queue_size.item())

        # Process completed acks
        while finish_count > 0:
            _, finish_event, ack_list = cc.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                node, params = self.ongoing_write_through.pop(ack_id)
                self._record_store_event(node, medium=StorageMedium.CPU)
                self.dec_lock_ref(node, params)
                if self.enable_storage:
                    self.write_backup_storage(node)
            finish_count -= 1
        for node_id in list(self.ongoing_fluxon_hostless_backup):
            self._finish_fluxon_hostless_write_batch(
                node_id,
                block=False,
                caller="writing_check(post_ack_write_queue)",
            )

    def loading_check(self) -> None:
        """Poll load-back completions."""
        cc = self.cache_controller
        if cc is None or not self.ongoing_load_back:
            return
        finish_count = 0
        for _, finish_event, ack_list in cc.ack_load_queue:
            operations = [
                operation
                for ack_id in ack_list
                if (
                    operation := self.ongoing_fluxon_hostless_layerwise_load.get(
                        ack_id
                    )
                )
                is not None
            ]
            submit_futures = {
                operation.submit_future
                for operation in operations
                if operation.submit_future is not None
            }
            # An unrecorded CUDA event reports ready. Never query the last
            # layer until the background thread has recorded every event.
            if any(not future.done() for future in submit_futures):
                break
            submit_error: BaseException | None = None
            for future in submit_futures:
                try:
                    future.result()
                except BaseException as exc:
                    submit_error = exc
                    break
            if submit_error is not None:
                try:
                    finish_event.synchronize()
                except BaseException as sync_exc:
                    logger.exception(
                        "Fluxon failed to synchronize after background DMA "
                        "submission error: submit_error=%s sync_error=%s",
                        submit_error,
                        sync_exc,
                    )
                for operation in operations:
                    self._abort_fluxon_hostless_layerwise_load(operation)
                del cc.ack_load_queue[: finish_count + 1]
                raise RuntimeError(
                    "Fluxon background DMA submission failed; restored pages "
                    "were rolled back"
                ) from submit_error
            if not finish_event.query():
                break
            finish_count += 1
            finish_event.synchronize()
            for ack_id in ack_list:
                operation = self.ongoing_fluxon_hostless_layerwise_load.pop(
                    ack_id,
                    None,
                )
                try:
                    if operation is not None:
                        restore_complete_ms = (
                            time.perf_counter() - operation.queued_at
                        ) * 1000.0
                        released_gpu_staging = (
                            operation.gpu_staging_lease is not None
                        )
                        try:
                            operation.release_views()
                        except Exception as exc:
                            logger.warning(
                                "Fluxon layerwise restore view release failed: "
                                "node=%d error=%s",
                                operation.node_id,
                                exc,
                            )
                        logger.info(
                            "Fluxon layerwise restore complete: node=%d tokens=%d "
                            "duration_ms=%.3f background_submit_cpu_ms=%s",
                            operation.node_id,
                            operation.token_count,
                            restore_complete_ms,
                            (
                                f"{operation.background_submit_cpu_ms:.3f}"
                                if operation.background_submit_cpu_ms is not None
                                else "none"
                            ),
                        )
                        operation.submit_future = None
                        operation.submit_guard = None
                        operation.submit_finish_event = None
                        operation.submit_stream = None
                        if (
                            operation.req_id
                            in self._fluxon_hostless_request_observations
                        ):
                            restore_complete_age_ms = (
                                self._fluxon_hostless_observation_age_ms(
                                    operation.req_id
                                )
                            )
                            self._finish_fluxon_hostless_request_observation(
                                operation.req_id,
                                "load_back_consumed",
                                restore_complete_ms=restore_complete_ms,
                                restore_complete_age_ms=(
                                    restore_complete_age_ms
                                ),
                                staging_release_age_ms=(
                                    restore_complete_age_ms
                                    if released_gpu_staging
                                    else 0.0
                                ),
                            )
                finally:
                    node, lock_params = self.ongoing_load_back.pop(ack_id)
                    self.dec_lock_ref(node, lock_params)
        del cc.ack_load_queue[:finish_count]

    # ---- HiCache: Scheduler Entry Points ----

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> tuple[torch.Tensor, UnifiedTreeNode]:
        """Prepare KV cache loading from host to device.
        Returns (device_indices, last_node) tuple."""
        best_match_node = params.best_match_node
        mem_quota = params.mem_quota
        req = params.req
        assert req is not None
        last_best_match_device_node = req.last_node
        restore_match_node = best_match_node
        partial_ready_restore = False
        if self._is_fluxon_hostless_full_mode():
            ready_operation = self.fluxon_hostless_ready_prefetch.get(req.rid)
            if ready_operation is not None and ready_operation.hash_value:
                ready_failure_shape: dict[str, Any] = {}
                ready_node = self._fluxon_hostless_longest_ready_restore_node(
                    ready_operation,
                    best_match_node,
                    last_best_match_device_node,
                    failure_shape=ready_failure_shape,
                )
                if ready_node is None:
                    logger.info(
                        "Fluxon ready restore failure shape: req=%s reason=%s "
                        "plan_anchor_node=%d current_anchor_node=%d "
                        "last_device_node=%d ready_pages=%d path_nodes=%d "
                        "path_pages=%d failure_node_index=%d "
                        "failure_node_pages=%d consumed_pages=%d "
                        "matched_pages=%d remaining_ready_pages=%d "
                        "plan_offset_pages=%d alignment_candidates=%d",
                        req.rid,
                        ready_failure_shape.get("reason", "unobserved"),
                        int(getattr(ready_operation, "anchor_node_id", -1)),
                        int(getattr(best_match_node, "id", -1)),
                        int(getattr(last_best_match_device_node, "id", -1)),
                        int(ready_failure_shape.get("ready_pages", 0)),
                        int(ready_failure_shape.get("path_nodes", 0)),
                        int(ready_failure_shape.get("path_pages", 0)),
                        int(ready_failure_shape.get("failure_node_index", -1)),
                        int(ready_failure_shape.get("failure_node_pages", 0)),
                        int(ready_failure_shape.get("consumed_pages", 0)),
                        int(ready_failure_shape.get("matched_pages", 0)),
                        int(ready_failure_shape.get("remaining_ready_pages", 0)),
                        int(ready_failure_shape.get("plan_offset_pages", 0)),
                        int(ready_failure_shape.get("alignment_candidates", 0)),
                    )
                    stale_operation = self.fluxon_hostless_ready_prefetch.pop(req.rid)
                    assert stale_operation is ready_operation, (
                        "Fluxon ready-prefetch identity changed during one scheduler turn"
                    )
                    self._cancel_fluxon_hostless_prefetch_operation(
                        stale_operation,
                        "ready_no_whole_node_prefix",
                    )
                    if req.rid in self._fluxon_hostless_request_observations:
                        self._finish_fluxon_hostless_request_observation(
                            req.rid,
                            "ready_no_whole_node_prefix",
                            anchor_node_id=best_match_node.id,
                            host_hit_tokens=int(params.host_hit_length),
                        )
                    return (
                        self._empty_match_result.device_indices,
                        last_best_match_device_node,
                    )
                if ready_operation.kv_plan_offset_pages > 0:
                    aligned_pages = len(
                        self._node_hash_values_after_ancestor(
                            ready_node,
                            last_best_match_device_node,
                        )
                    )
                    logger.info(
                        "Fluxon ready restore suffix aligned: req=%s "
                        "plan_anchor_node=%d current_anchor_node=%d "
                        "last_device_node=%d plan_offset_pages=%d "
                        "aligned_pages=%d ready_pages=%d",
                        req.rid,
                        int(getattr(ready_operation, "anchor_node_id", -1)),
                        int(getattr(best_match_node, "id", -1)),
                        int(getattr(last_best_match_device_node, "id", -1)),
                        int(ready_operation.kv_plan_offset_pages),
                        aligned_pages,
                        min(
                            len(ready_operation.hash_value),
                            ready_operation.completed_tokens // self.page_size,
                        ),
                    )
                restore_match_node = ready_node
                partial_ready_restore = restore_match_node is not best_match_node

        def _collect_new_prefix_indices() -> torch.Tensor:
            prefix_chunks: list[torch.Tensor] = []
            node = restore_match_node
            while node is not last_best_match_device_node:
                value = node.component_data[BASE_COMPONENT_TYPE].value
                assert value is not None
                prefix_chunks.append(value)
                node = node.parent
            if not prefix_chunks:
                return self._empty_match_result.device_indices
            prefix_chunks.reverse()
            return torch.cat(prefix_chunks)

        if restore_match_node.evicted or params.host_hit_length > 0:
            if self.load_back(restore_match_node, mem_quota, req=req):
                new_indices = _collect_new_prefix_indices()
                if new_indices.numel() == 0:
                    return (
                        self._empty_match_result.device_indices,
                        last_best_match_device_node,
                    )
                if partial_ready_restore:
                    restored_tokens = len(new_indices)
                    req.host_hit_length = restored_tokens
                    req.storage_hit_length = restored_tokens
                return new_indices, restore_match_node

        return (
            self._empty_match_result.device_indices,
            last_best_match_device_node,
        )

    def check_hicache_events(self) -> None:
        """Called per scheduler step to poll async HiCache events."""
        self.writing_check()
        self.loading_check()
        self._drain_fluxon_hostless_acks(block=False)
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def flush_write_through_acks(self) -> None:
        """Flush pending write-through acknowledgements."""
        self.writing_check()

    def ready_to_load_host_cache(self) -> int:
        """Notify the cache controller to start the KV cache loading."""
        if self._is_fluxon_hostless_full_mode():
            return self._start_fluxon_hostless_layerwise_loads()
        if self.cache_controller is not None:
            return self.cache_controller.start_loading()
        return 0

    # ---- Query / Inspection APIs ----
    # These APIs exist for compatibility with other RadixTree implementations.
    # TODO: simplify and consolidate in a future refactor.

    @property
    def sliding_window_size(self):
        swa = self.components.get(ComponentType.SWA)
        return swa.sliding_window_size if swa else None

    def supports_swa(self) -> bool:
        return ComponentType.SWA in self.components

    def supports_mamba(self) -> bool:
        return ComponentType.MAMBA in self.components

    # ---- Streaming session API (delegates to composed StreamingSession) ----

    def supports_streaming_session(self) -> bool:
        return True

    def release_session(self, session_id: str) -> None:
        self.session.release_session(session_id)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_tokens(active_pool_idxs)

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_full_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_swa_tokens(active_pool_idxs)

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_req_count(active_pool_idxs)

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_mamba_slots(active_pool_idxs)

    def evictable_size(self) -> int:
        return self.component_evictable_size_.get(BASE_COMPONENT_TYPE, 0)

    def protected_size(self) -> int:
        return self.component_protected_size_.get(BASE_COMPONENT_TYPE, 0)

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.SWA, 0)

    def mamba_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.MAMBA, 0)

    def swa_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.SWA, 0)

    def mamba_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.MAMBA, 0)

    def total_size(self):
        total_size = 0
        total_aux_size = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            if full_value is not None:
                total_size += len(full_value)
            for ct in self.tree_components:
                if ct == BASE_COMPONENT_TYPE:
                    continue
                value = node.component_data[ct].value
                if value is not None:
                    total_aux_size += len(value)
            for child in node.children.values():
                stack.append(child)
        return total_size, total_aux_size

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs(node: UnifiedTreeNode):
            for child in node.children.values():
                v = child.component_data[BASE_COMPONENT_TYPE].value
                if v is not None:
                    values.append(v)
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def _all_component_values_flatten(
        self, component_type: ComponentType
    ) -> torch.Tensor:
        if component_type not in self.components:
            return torch.tensor([], dtype=torch.int64, device=self.device)

        values = []

        def _dfs(node: UnifiedTreeNode):
            value = node.component_data[component_type].value
            if value is not None:
                values.append(value)
            for child in node.children.values():
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentType.MAMBA)

    def all_swa_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentType.SWA)

    def available_and_evictable_str(self) -> str:
        if self.supports_swa():
            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        else:
            full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable = self.component_evictable_size_[BASE_COMPONENT_TYPE]
        lines = [
            f"Available full tokens: {full_available_size + full_evictable} "
            f"(full_available_size={full_available_size} + full_evictable_size_={full_evictable})"
        ]
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            if ct.is_swa:
                available_size = self.token_to_kv_pool_allocator.swa_available_size()
            elif ct.is_mamba:
                available_size = self.req_to_token_pool.mamba_pool.available_size()
            else:
                continue

            lines.append(
                f"Available {ct}: {available_size + self.component_evictable_size_[ct]} "
                f"(available_size={available_size} + component_evictable_size_={self.component_evictable_size_[ct]})"
            )
        return "\n".join(lines) + "\n"

    def _collect_all_nodes(self) -> list[UnifiedTreeNode]:
        nodes = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        return nodes

    def sanity_check(self):
        """Verify tree invariants.

        TODO(hzh): This method has relatively high latency; simplify the
        check logic once the tree implementation stabilizes.
        """
        # Skip when streaming sessions hold tree locks: the check asserts
        # all nodes are unlocked during idle, which streaming sessions break
        # by design (they hold a first-turn lock across turns).
        if self.session.any_holding_kv():
            return

        write_back = (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        )

        errors: list[str] = []
        E = errors.append
        all_nodes = self._collect_all_nodes()
        all_node_set = set(all_nodes)
        FCT = BASE_COMPONENT_TYPE

        # ── PART 1: Tree Structure ──
        # Root state
        if self.root_node.component_data[FCT].value is None:
            E("[Root] root missing Full device value")
        if self.root_node.component_data[FCT].lock_ref <= 0:
            E(
                f"[Root] root Full lock_ref={self.root_node.component_data[FCT].lock_ref}"
            )
        if self.root_node.parent is not None:
            E("[Root] root has a parent pointer")
        # Parent ↔ child bidirectional consistency
        for node in all_nodes:
            for child in node.children.values():
                if child.parent is not node:
                    pid = child.parent.id if child.parent else None
                    E(f"[Tree] child {child.id} parent={pid}, expected {node.id}")
                if child.key is None:
                    E(f"[Tree] node {child.id} has no key")

        # ── PART 2: Per-node state machine and leaf qualification ──
        expected_dev_leaves: set[UnifiedTreeNode] = set()
        expected_hst_leaves: set[UnifiedTreeNode] = set()

        for node in all_nodes:
            if node is self.root_node:
                continue
            nid = node.id
            full_dev = node.component_data[FCT].value is not None
            full_hst = node.component_data[FCT].host_value is not None
            full_storage = self._is_component_storage_backed(
                node.component_data[FCT]
            )
            full_lower = full_hst or full_storage

            # Full is the tree backbone, so aux data requires Full data.
            for ct in self.tree_components:
                if ct == FCT:
                    continue
                cd = node.component_data[ct]
                if cd.value is not None and not full_dev:
                    E(f"node {nid} {ct} device present but Full.value=None")
                if cd.host_value is not None and not full_hst:
                    E(f"node {nid} {ct} host present but Full.host_value=None")

            # Every node must keep Full data on at least one layer.
            if not full_dev and not full_lower:
                E(f"node {nid} dead: no Full device and no lower-tier backup")

            # Parent prefixes must keep data whenever the child does.
            if node.parent is not None and node.parent is not self.root_node:
                p_dev = node.parent.component_data[FCT].value is not None
                p_hst = node.parent.component_data[FCT].host_value is not None
                p_storage = self._is_component_storage_backed(
                    node.parent.component_data[FCT]
                )
                if full_dev and not p_dev:
                    E(f"node {nid} device present but parent {node.parent.id} evicted")
                if full_lower and not (p_hst or p_storage) and not write_back:
                    E(f"node {nid} backed up but parent {node.parent.id} not backed up")

            # Lock hierarchy and counters must stay sane.
            fl = node.component_data[FCT].lock_ref
            for ct in self.tree_components:
                cd = node.component_data[ct]
                if cd.lock_ref < 0:
                    E(f"node {nid} {ct} lock_ref={cd.lock_ref}")
                if cd.host_lock_ref < 0:
                    E(f"node {nid} {ct} host_lock_ref={cd.host_lock_ref}")
                if ct != FCT and fl < cd.lock_ref:
                    E(f"node {nid} full_lock={fl} < {ct}_lock={cd.lock_ref}")
                if cd.value is None and cd.lock_ref > 0:
                    E(f"node {nid} {ct} evicted but lock_ref={cd.lock_ref}")

            # Collect expected leaf qualification (single pass)
            if self._is_device_leaf(node):
                expected_dev_leaves.add(node)
            if self._is_host_leaf(node):
                expected_hst_leaves.add(node)

        # ── PART 3: Tracking structures ──

        # Device leaf set must match the expected leaves.
        if self.evictable_device_leaves != expected_dev_leaves:
            extra = self.evictable_device_leaves - expected_dev_leaves
            missing = expected_dev_leaves - self.evictable_device_leaves
            if extra:
                E(f"D-leaf extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"D-leaf missing: {[n.id for n in list(missing)[:5]]}")

        # Host leaf set must match the expected leaves.
        if self.evictable_host_leaves != expected_hst_leaves:
            extra = self.evictable_host_leaves - expected_hst_leaves
            missing = expected_hst_leaves - self.evictable_host_leaves
            if extra:
                E(f"H-leaf extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"H-leaf missing: {[n.id for n in list(missing)[:5]]}")

        # D-leaf ∩ H-leaf = ∅
        overlap = self.evictable_device_leaves & self.evictable_host_leaves
        if overlap:
            E(
                f"[Leaf] {len(overlap)} in both sets: {[n.id for n in list(overlap)[:5]]}"
            )

        # Stale nodes: leaf sets must only contain tree-reachable nodes
        stale = self.evictable_device_leaves - all_node_set
        if stale:
            E(
                f"{len(stale)} stale nodes in device_leaves: {[n.id for n in list(stale)[:5]]}"
            )
        stale = self.evictable_host_leaves - all_node_set
        if stale:
            E(
                f"{len(stale)} stale nodes in host_leaves: {[n.id for n in list(stale)[:5]]}"
            )

        # Per-component LRU tracking
        for ct in self.tree_components:
            lru = self.lru_lists[ct]
            if ct == FCT:
                # Full uses leaf sets, not LRU
                if len(lru.cache) > 0:
                    E(f"Full device LRU not empty: {len(lru.cache)}")
                if len(self.host_lru_lists[ct].cache) > 0:
                    E(f"Full host LRU not empty: {len(self.host_lru_lists[ct].cache)}")
            else:
                # Aux device values must match the device LRU.
                tree_ids = {
                    n.id
                    for n in all_nodes
                    if n is not self.root_node
                    and n.component_data[ct].value is not None
                }
                lru_ids = set(lru.cache.keys())
                if tree_ids != lru_ids:
                    E(
                        f"{ct} device LRU: "
                        f"+tree={tree_ids - lru_ids}, +lru={lru_ids - tree_ids}"
                    )
                # Aux host-only states must match the host LRU.
                host_lru = self.host_lru_lists[ct]
                s3_ids = {
                    n.id
                    for n in all_nodes
                    if n is not self.root_node
                    and n.component_data[ct].value is None
                    and n.component_data[ct].host_value is not None
                }
                host_lru_ids = set(host_lru.cache.keys())
                if s3_ids != host_lru_ids:
                    E(
                        f"{ct} host LRU: "
                        f"+S3={s3_ids - host_lru_ids}, +lru={host_lru_ids - s3_ids}"
                    )
                # The same aux node must not appear in both device and host LRU.
                inv5_overlap = lru_ids & host_lru_ids
                if inv5_overlap:
                    E(f"{ct} in both device and host LRU: {inv5_overlap}")
                # Linked-list integrity
                self._check_lru_linked_list(lru, ct, "device", errors)
                self._check_lru_linked_list(host_lru, ct, "host", errors)

        # ── PART 4: Size Accounting ──
        for ct in self.tree_components:
            evictable = 0
            protected = 0
            for n in all_nodes:
                if n is self.root_node:
                    continue
                cd = n.component_data[ct]
                if cd.value is not None:
                    toks = len(cd.value)
                    if cd.lock_ref > 0:
                        protected += toks
                    else:
                        evictable += toks
            if self.component_evictable_size_[ct] != evictable:
                E(
                    f"[Size] {ct} evictable={self.component_evictable_size_[ct]} "
                    f"!= recomputed={evictable}"
                )
            if self.component_protected_size_[ct] != protected:
                E(
                    f"[Size] {ct} protected={self.component_protected_size_[ct]} "
                    f"!= recomputed={protected}"
                )

        # ── PART 5: Ongoing Operations ──
        for nid, (n, _) in self.ongoing_write_through.items():
            if n not in all_node_set:
                E(f"[Ongoing] write_through node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] write_through node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )
        for nid, (n, _) in self.ongoing_load_back.items():
            if n not in all_node_set:
                E(f"[Ongoing] load_back node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] load_back node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )
        for nid, batch in self.ongoing_fluxon_hostless_backup.items():
            n = batch.node
            if n not in all_node_set:
                E(f"[Ongoing] fluxon_hostless_backup node {nid} not in tree")
            if not bool(n.component_data[FCT].metadata.get("storage_pending", False)):
                E(f"[Ongoing] fluxon_hostless_backup node {nid} missing storage_pending")
            if not bool(n.component_data[FCT].metadata.get("storage_staged", False)):
                E(f"[Ongoing] fluxon_hostless_backup node {nid} missing storage_staged")

        # ── Result ──
        if errors:
            msg = (
                f"Sanity check FAILED ({len(errors)} violations "
                f"across {len(all_nodes)} nodes):\n"
                + "\n".join(f"  {e}" for e in errors)
            )
            logger.error(msg)
            self.pretty_print()
            raise AssertionError(msg)

    def _check_lru_linked_list(
        self,
        lru: "UnifiedLRUList",
        ct: ComponentType,
        label: str,
        errors: list[str],
    ) -> None:
        """Walk a LRU doubly-linked list, collect integrity errors."""
        pt = lru._pt  # use LRU's own pointer slot
        visited: set[int] = set()
        x = lru.head.lru_next[pt]
        prev = lru.head
        while x is not None and x != lru.tail:
            if x.lru_prev[pt] != prev:
                errors.append(f"[{label}][{ct}] broken prev at node {x.id}")
            if x.id not in lru.cache:
                errors.append(f"[{label}][{ct}] node {x.id} in list not cache")
            if x.id in visited:
                errors.append(f"[{label}][{ct}] cycle at node {x.id}")
                break
            visited.add(x.id)
            prev = x
            x = x.lru_next[pt]
        if x is None:
            errors.append(
                f"[{label}][{ct}] broken chain: lru_next is None "
                f"after node {prev.id if hasattr(prev, 'id') else 'head'}"
            )
        if len(visited) != len(lru.cache):
            errors.append(
                f"[{label}][{ct}] list={len(visited)} != cache={len(lru.cache)}"
            )

    def pretty_print(self) -> None:
        stack = [(self.root_node, 0)]
        while stack:
            node, indent = stack.pop()
            component_str = " ".join(
                f"{ct}={'yes' if node.component_data[ct].value is not None else 'no'}"
                for ct in self.tree_components
            )
            print(
                " " * indent,
                f"[{node.id}]",
                len(node.key),
                f"full_lock={node.component_data[BASE_COMPONENT_TYPE].lock_ref}",
                component_str,
            )
            for child in node.children.values():
                stack.append((child, indent + 2))

    def _rebuild_host_leaf_sets(self) -> None:
        """Rebuild evictable_host_leaves after L1-only reset."""
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node is not self.root_node:
                self._update_evictable_leaf_sets(node)
            stack.extend(node.children.values())

    def _rebuild_host_lru_lists(self) -> None:
        """Rebuild host_lru_lists for extra components after L1-only reset.
        Walks the tree and adds nodes with host component data to the
        appropriate host LRU list."""
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node is not self.root_node:
                for ct in self.tree_components:
                    if ct == BASE_COMPONENT_TYPE:
                        continue  # Full uses evictable_host_leaves, not host LRU
                    cd = node.component_data[ct]
                    if cd.host_value is not None:
                        self.host_lru_lists[ct].insert_mru(node)
            stack.extend(node.children.values())
