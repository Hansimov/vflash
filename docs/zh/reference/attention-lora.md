# 可选 FP32 注意力 LoRA

原生上下文和完整 Python/CLI 管线均可为单 SM89、串行的官方 H3 Base16 运行时附加
仅 DiT 主干的注意力残差，默认关闭。不改准备资源、不合并底模、不下载权重或自动选择模型。

## 生成完整视频

把本地 adapter 显式传给拥有编码、去噪和解码的完整管线：

```python
from pathlib import Path
from vflash.pipeline import AttentionAdapter, H3Pipeline

with H3Pipeline(
    prepared, device=device, trust_local_code=True,
    attention_adapter=AttentionAdapter(Path("attention-adapter.safetensors"), rank=8, scale=-1),
) as pipeline:
    result = pipeline.generate(request, Path("result.mp4"))
```

`vflash generate` 同时传入三个参数：
`--attention-adapter attention-adapter.safetensors --attention-adapter-rank 8 --attention-adapter-scale -1`。
使用 Base16 I2VA/FL2VA 准备资源，L2VA 沿用其关键帧 profile 合同。
该 adapter 尚不支持 SM86 或协同双卡；HTTP 配置不变。
准备时只加载一次，连续请求复用；正常关闭或失败时先卸载 adapter，再关闭原生核心。
结果的 `stages.denoising.attention_adapter` 标明范围、精度、rank 和倍率，调用方还需保留权重身份。
省略该配置即可回到原模型，不修改任何底模文件。

管线包装的 CPU 生命周期检查已通过，新完整管线 GPU 验证待执行；下方既有原生实测不冒充包装验证。

## 在已有原生 session 中使用

按 [Python 集成](../guide/python) 创建 `i2va-base16-bf16-sm89` session，使用对应资源；
Sol 使用 block-ring。随后：

```python
from pathlib import Path
from safetensors.torch import load_file
from vflash.native.h3_attention_lora import apply_dit_attention_lora

state = load_file("attention-adapter.safetensors", device="cpu")
# 位于独占、已启动的 Base16 NativeEngineSession 内：
with apply_dit_attention_lora(session.runtime, state, rank=8, scale=-1.0) as calls:
    result = session.generate(
        Path("bundles/example"), Path("outputs/adapter-latents.safetensors")
    )
print(calls)
# 退出后，同一 session 恢复使用原底模。
```

输出仍是 AV latent，需匹配的官方解码器生成 MP4。结果旁应记录权重身份、rank、有效
倍率、仅主干范围和引擎版本，不能因底模 artifact 未改就把输出标为无 adapter 的基线。

## 格式与精度

输入为完整 208 个 CPU FP32 PEFT 注意力张量：50 个 DiT block 和两个 TokenRefiner
block 的 QKV/output A/B。文本层只验证，**不应用**；这不等同于另一框架的完整 adapter。

QKV B 按 head 交错的行会转成完整 Q/K/V 排序。残差的两次 GEMM 和相加使用 FP32，
随后返回底模激活类型，不将小更新预先舍入合入 BF16 权重。按行分块限制临时显存；
环形缓冲在计算线程绑定逻辑层，不在超前预取中错误地选择下一层残差。

`scale` 必须显式传入 [-1, 1]，包含 alpha/rank 因子。alpha=rank 时，-1 表示减去原更新
一次；B 已经取负时不要再传 -1。加载前检查形状、精度和有限值。
SM86、多 GPU、其他步数、已有编译 adapter 以及嵌套上下文均拒绝。

可选权重示例为 [反向注意力 adapter](https://huggingface.co/rockstarengine/vflash)。
遵守其模型卡和 MiniMax 模型许可，不能与本包源码许可混同。本包不携带权重或私有媒体。

## 生命周期与证据边界

运行时/进程独占，串行请求，上下文期间不修改输入权重。正常退出和异常退出都会恢复
原方法；CUDA 请求失败后仍应关闭运行时，不能仅凭方法恢复就假定设备状态可复用。

聚焦检查覆盖 FP32 运算、QKV 排序、环形逻辑层、重复调用及异常清理。单 RTX 4090 48 GB
固定 672×384、6 秒、24fps、Base16/Sol block-ring 已完整生成，公开提取的 Base 和反向
两版 AV latent 分别与旧实现逐值一致。这验证代码迁移，不代表 Base 与反向画面相同。
此前机制另有 864×864/10 秒及独立留出场景证据，本次提取未重跑大画布。它不保证所有脸部、
动作或声音改善，残余伪影仍在，构图也可能改变。按原始需求判断，不以匹配 Base 画面
作为质量标准。接口不自动在生产启用，也不改变 Base16/Sol 默认。
