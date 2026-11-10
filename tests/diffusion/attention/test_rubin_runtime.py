# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise the encoder, DiT attention, and compiler paths used on Rubin."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "Rubin"}, num_cards=1),
]

if not current_platform.is_cuda():
    pytest.skip("Rubin runtime tests require NVIDIA CUDA", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def require_rubin() -> None:
    if current_platform.get_device_capability() != (10, 7):
        pytest.skip("Rubin runtime tests require compute capability 10.7")


@pytest.mark.parametrize("length", [256, 1024])
def test_rubin_vision_sdpa_with_strided_value(length: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(123)
    packed = torch.randn(length, 3, 16, 72, generator=generator, device="cuda", dtype=torch.bfloat16)
    query, key, value = packed.permute(1, 0, 2, 3).unbind(0)
    query = query.contiguous().transpose(0, 1).unsqueeze(0)
    key = key.contiguous().transpose(0, 1).unsqueeze(0)
    # The H3 vision encoder keeps V strided after splitting the QKV projection.
    value = value.transpose(0, 1).unsqueeze(0)

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        actual = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.02)


@pytest.mark.parametrize("valid_kv_length", [1024, 1007])
def test_rubin_dit_attention_with_valid_kv_prefix(valid_kv_length: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    query, key, value = [
        torch.randn(1, 1024, 6, 128, generator=generator, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    scale = 128**-0.5
    impl = CuDNNAttentionImpl(num_heads=6, head_size=128, softmax_scale=scale, backend_explicit=True)

    actual = impl.forward_cuda(query, key, value, AttentionMetadata(extra={"valid_kv_length": valid_kv_length}))
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key[:, :valid_kv_length].transpose(1, 2),
            value[:, :valid_kv_length].transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        ).transpose(1, 2)

    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.02)


@triton.jit
def _rms_kernel(x_ptr, y_ptr, width: tl.constexpr, epsilon: tl.constexpr):
    row = tl.program_id(0)
    offsets = row * width + tl.arange(0, width)
    values = tl.load(x_ptr + offsets).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / width
    tl.store(y_ptr + offsets, values * tl.rsqrt(mean_square + epsilon))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [128, 1024])
def test_rubin_triton_reduction_emits_native_ptx(dtype: torch.dtype, width: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(17)
    values = torch.randn(8, width, generator=generator, device="cuda", dtype=dtype)
    actual = torch.empty_like(values)
    epsilon = 1e-6

    kernel = _rms_kernel[(values.shape[0],)](values, actual, width, epsilon, num_warps=4)
    expected = values.float() * torch.rsqrt(values.float().square().mean(dim=-1, keepdim=True) + epsilon)

    tolerance = 0.02 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(actual, expected.to(dtype), atol=tolerance, rtol=tolerance)
    assert ".target sm_107a" in kernel.asm["ptx"]
