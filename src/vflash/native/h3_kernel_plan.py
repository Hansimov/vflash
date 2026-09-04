"""Fixed BF16 and strict-fusion kernel plans for SM86 and SM89."""

from __future__ import annotations

from dataclasses import dataclass


class H3KernelPlanError(ValueError):
    """No measured H3 kernel portfolio matches the requested target."""


@dataclass(frozen=True)
class H3KernelPlan:
    plan_id: str
    compute_capability: str
    quality_contract: str
    weight_residency: str
    main_gemm_backend: str
    attention_backend: str
    elementwise_backend: str
    elementwise_block_size: int
    adapter_fusion_backend: str
    rotary_backend: str
    adapter_execution: str
    prompt_conditioner_backend: str
    input_projection_backend: str
    input_scatter_backend: str
    final_adaln_backend: str
    final_norm_backend: str
    final_output_head_backend: str
    float32_matmul_precision: str
    row_timestep_backend: str
    scheduler_backend: str
    latent_state_backend: str


_STRICT_PLANS = {
    "sm86": H3KernelPlan(
        plan_id="h3-sm86-strict-bf16-ring-v5",
        compute_capability="sm86",
        quality_contract="official-bf16-and-tf32-rounding-boundaries",
        weight_residency="pinned-host-two-slot-event-ring",
        main_gemm_backend="cublas-bf16-tensor-core",
        attention_backend="torch-flash-sdpa",
        elementwise_backend="triton-strict",
        elementwise_block_size=512,
        adapter_fusion_backend="triton-strict",
        rotary_backend="triton-strict",
        adapter_execution="runtime-residual-compatible",
        prompt_conditioner_backend="cublas-bf16-torch-flash-prompt-cache",
        input_projection_backend="cublas-tf32-fp32-accumulate",
        input_scatter_backend="torch-index-copy-inplace-bf16",
        final_adaln_backend="precomputed-cublas-bf16",
        final_norm_backend="torch-rms-norm-bf16",
        final_output_head_backend="cublas-tf32-fp32-accumulate",
        float32_matmul_precision="high",
        row_timestep_backend="precomputed-torch-fp32-unique",
        scheduler_backend="torch-fp32-training-euler-exact-order",
        latent_state_backend="stable-address-generated-suffix-inplace",
    ),
    "sm89": H3KernelPlan(
        plan_id="h3-sm89-strict-bf16-resident-v5",
        compute_capability="sm89",
        quality_contract="official-bf16-and-tf32-rounding-boundaries",
        weight_residency="device-resident",
        main_gemm_backend="cublas-bf16-tensor-core",
        attention_backend="torch-flash-sdpa",
        elementwise_backend="triton-strict",
        elementwise_block_size=1024,
        adapter_fusion_backend="triton-strict",
        rotary_backend="triton-strict",
        adapter_execution="runtime-residual-compatible",
        prompt_conditioner_backend="cublas-bf16-torch-flash-prompt-cache",
        input_projection_backend="cublas-tf32-fp32-accumulate",
        input_scatter_backend="torch-index-copy-inplace-bf16",
        final_adaln_backend="precomputed-cublas-bf16",
        final_norm_backend="torch-rms-norm-bf16",
        final_output_head_backend="cublas-tf32-fp32-accumulate",
        float32_matmul_precision="high",
        row_timestep_backend="precomputed-torch-fp32-unique",
        scheduler_backend="torch-fp32-training-euler-exact-order",
        latent_state_backend="stable-address-generated-suffix-inplace",
    ),
}


def resolve_h3_kernel_plan(compute_capability: str) -> H3KernelPlan:
    """Return the measured strict-quality portfolio for one target SM."""

    try:
        return _STRICT_PLANS[compute_capability]
    except KeyError as exc:
        raise H3KernelPlanError(
            f"no measured strict H3 kernel plan for {compute_capability}"
        ) from exc
