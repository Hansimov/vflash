# 可选 FP32 注意力 LoRA

这是默认关闭的实验性**底层接口**，为单 SM89、串行的官方 H3 Base16 运行时临时附加
仅 DiT 主干的注意力残差。不改准备资源、不合并底模、不自动选择模型，也不新增
pipeline/HTTP 参数。

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

聚焦检查覆盖 FP32 运算、QKV 排序、环形逻辑层、重复调用及异常清理。此前机制已有
单 SM89 小/大画布成片证据，公开提取版本正进行同输入迁移核验。它不保证所有脸部、
动作或声音改善，残余伪影仍在，构图也可能改变。按原始需求判断，不以匹配 Base 画面
作为质量标准。接口不自动在生产启用，也不改变 Base16/Sol 默认。

