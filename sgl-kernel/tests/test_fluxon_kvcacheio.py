import pytest
import torch

from sgl_kernel.kvcacheio import (
    restore_mamba_state_from_fluxon_values,
    restore_mha_pages_from_fluxon_values,
    restore_mla_pages_from_fluxon_values,
    transfer_raw_h2d_batch,
    write_mamba_state_to_fluxon_values,
    write_mha_pages_to_fluxon_values,
    write_mla_pages_to_fluxon_values,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Fluxon KV transfer tests require CUDA"
)

_PLAN_MAGIC = 0x4658504C414E5631


def _pointer_tensor(tensors):
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        dtype=torch.int64,
        device="cuda",
    )


def _plan_blob(values):
    blob = torch.empty(len(values) + 2, dtype=torch.int64, pin_memory=True)
    blob[0] = _PLAN_MAGIC
    blob[1] = len(values)
    blob[2:] = torch.tensor([value.data_ptr() for value in values])
    return blob


def test_transfer_raw_h2d_batch():
    source_a = torch.arange(31, dtype=torch.uint8, pin_memory=True)
    source_b = torch.arange(17, dtype=torch.uint8, pin_memory=True) + 41
    destination_a = torch.zeros_like(source_a, device="cuda")
    destination_b = torch.zeros_like(source_b, device="cuda")

    transfer_raw_h2d_batch(
        torch.tensor([destination_a.data_ptr(), destination_b.data_ptr()]),
        torch.tensor([source_a.data_ptr(), source_b.data_ptr()]),
        torch.tensor([source_a.nbytes, source_b.nbytes]),
        torch.cuda.current_device(),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(destination_a.cpu(), source_a)
    torch.testing.assert_close(destination_b.cpu(), source_b)


def test_fluxon_mha_page_round_trip():
    page_count = 3
    page_bytes = 16
    layers = 2
    page_indices = torch.tensor([2, 0], dtype=torch.int64, device="cuda")
    k_source = [
        (
            torch.arange(page_count * page_bytes, device="cuda", dtype=torch.uint8)
            + 17 * layer
        )
        for layer in range(layers)
    ]
    v_source = [
        (
            torch.arange(page_count * page_bytes, device="cuda", dtype=torch.uint8)
            + 71
            + 13 * layer
        )
        for layer in range(layers)
    ]
    values = [
        torch.empty(2 * layers * page_bytes, dtype=torch.uint8, pin_memory=True)
        for _ in range(page_indices.numel())
    ]
    plan = _plan_blob(values)

    write_mha_pages_to_fluxon_values(
        plan.data_ptr(),
        page_indices,
        _pointer_tensor(k_source),
        _pointer_tensor(v_source),
        page_bytes,
        page_bytes,
        torch.cuda.current_device(),
    )
    torch.cuda.synchronize()

    for output_index, page_index in enumerate(page_indices.cpu().tolist()):
        expected = torch.cat(
            [
                layer[page_index * page_bytes : (page_index + 1) * page_bytes].cpu()
                for layer in k_source
            ]
            + [
                layer[page_index * page_bytes : (page_index + 1) * page_bytes].cpu()
                for layer in v_source
            ]
        )
        torch.testing.assert_close(values[output_index], expected)

    k_restored = [torch.zeros_like(layer) for layer in k_source]
    v_restored = [torch.zeros_like(layer) for layer in v_source]
    restore_mha_pages_from_fluxon_values(
        plan.data_ptr(),
        page_indices,
        _pointer_tensor(k_restored),
        _pointer_tensor(v_restored),
        page_bytes,
        page_bytes,
        torch.cuda.current_device(),
    )
    torch.cuda.synchronize()

    for page_index in page_indices.cpu().tolist():
        page_slice = slice(page_index * page_bytes, (page_index + 1) * page_bytes)
        for restored, source in zip(k_restored + v_restored, k_source + v_source):
            torch.testing.assert_close(restored[page_slice], source[page_slice])


def test_fluxon_mla_page_round_trip():
    page_bytes = 24
    page_indices = torch.tensor([1, 0], dtype=torch.int64, device="cuda")
    source = [
        torch.arange(2 * page_bytes, dtype=torch.uint8, device="cuda") + layer * 29
        for layer in range(3)
    ]
    values = [
        torch.empty(len(source) * page_bytes, dtype=torch.uint8, pin_memory=True)
        for _ in range(page_indices.numel())
    ]
    plan = _plan_blob(values)

    write_mla_pages_to_fluxon_values(
        plan.data_ptr(),
        page_indices,
        _pointer_tensor(source),
        page_bytes,
        torch.cuda.current_device(),
    )
    torch.cuda.synchronize()

    restored = [torch.zeros_like(layer) for layer in source]
    restore_mla_pages_from_fluxon_values(
        plan.data_ptr(),
        page_indices,
        _pointer_tensor(restored),
        page_bytes,
        torch.cuda.current_device(),
    )
    torch.cuda.synchronize()
    for page_index in page_indices.cpu().tolist():
        page_slice = slice(page_index * page_bytes, (page_index + 1) * page_bytes)
        for restored_layer, source_layer in zip(restored, source):
            torch.testing.assert_close(
                restored_layer[page_slice], source_layer[page_slice]
            )


def test_fluxon_mamba_state_round_trip():
    layer_num = 2
    slot_count = 3
    slot_index = 1
    item_sizes = [8, 12]
    source_states = [
        torch.arange(
            layer_num * slot_count * item_size,
            dtype=torch.uint8,
            device="cuda",
        ).reshape(layer_num, slot_count, item_size)
        + state_index * 61
        for state_index, item_size in enumerate(item_sizes)
    ]
    value = torch.empty(
        sum(layer_num * item_size for item_size in item_sizes),
        dtype=torch.uint8,
        pin_memory=True,
    )
    plan = _plan_blob([value])
    state_item_bytes = torch.tensor(item_sizes, dtype=torch.int64, device="cuda")
    source_ptrs = _pointer_tensor(
        [state[layer] for state in source_states for layer in range(layer_num)]
    )

    write_mamba_state_to_fluxon_values(
        plan.data_ptr(),
        slot_index,
        source_ptrs,
        state_item_bytes,
        layer_num,
        torch.cuda.current_device(),
    )
    torch.cuda.synchronize()

    restored_states = [torch.zeros_like(state) for state in source_states]
    restored_ptrs = _pointer_tensor(
        [state[layer] for state in restored_states for layer in range(layer_num)]
    )
    restore_mamba_state_from_fluxon_values(
        plan.data_ptr(),
        slot_index,
        restored_ptrs,
        state_item_bytes,
        layer_num,
        torch.cuda.current_device(),
    )
    torch.cuda.synchronize()

    for restored, source in zip(restored_states, source_states):
        torch.testing.assert_close(restored[:, slot_index], source[:, slot_index])
