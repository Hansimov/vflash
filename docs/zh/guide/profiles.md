# 配置与硬件

一个运行配置（profile）确定模型、LoRA 版本、步数和计算方式；默认显存策略取决于所选硬件。选择时，需要让配置、硬件和编译资源相互匹配。

## 可用配置 {#available}

当前配置都从预编译条件包执行 Ref2VA 去噪。Ref2VA 是 H3 基于参考素材生成视频和音频的模式。

| 显卡 | 配置 | 步数 | 权重加载方式 |
| --- | --- | ---: | --- |
| RTX 4090 48 GB | `ref2va-turbo4-exact-sm89` | 4 | 常驻显存 |
| RTX 4090 48 GB | `ref2va-turbo8-exact-sm89` | 8 | 常驻显存 |
| RTX 3080 20 GB | `ref2va-turbo4-exact-sm86` | 4 | 从系统内存分块加载 |

HTTP 服务默认使用 4090 Turbo4。切换配置时，需要同时更换匹配的资源并重启服务。

```bash
vflash profiles
vflash plan ref2va-turbo8-exact-sm89 --gpu 0
```

## 内存与部署 {#memory}

| 硬件 | 显存策略 | 选择时考虑 |
| --- | --- | --- |
| 单 4090 48 GB | 默认权重常驻 | 多次请求复用权重；为中间结果留出足够显存 |
| 单或双 3080 20 GB | 从系统内存分块加载 | 权重保存在系统内存；双卡共同处理一个请求 |
| 单 4090，显存需要更多余量 | 显式选择 `block-ring` | 用系统内存换取显存余量，需要单独测量延迟 |

CLI 使用 `--weight-residency block-ring`，Python 使用同名的[会话选项](./python#memory)。HTTP 服务使用所选配置的默认策略。

已测的 928 × 512、124 模型帧、4 步负载，建议**每个 worker 至少预留 64 GiB 可用系统内存**，并为操作系统和其他进程留出余量。使用分块加载时，显存占用不包括系统内存中的完整权重。更大输入和并发 worker 都需要重新检查容量。

双 3080 可选择 `sequence-head` 或 `tensor`，由调用方显式指定第二张卡。已测拓扑为无 peer access 的 PCIe 3.0 x16 主机桥连接，不需要 NVLink。设置方法见[双卡执行](./getting-started#parallel)，对照范围见[实测数据](../reference/benchmarks#sm86-parallel)。

上述支持范围只覆盖 20 GB 版 3080 和 48 GB 版 4090。其他显存容量、型号、更大的 GPU 组，以及任意分辨率或帧数组合尚未获得相同验证。

## Turbo LoRA 支持 {#lora}

Vflash 可以直接执行固定版本的 [LightX2V H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) LoRA，无需安装 LightX2V 推理框架。配置中的 LoRA、调度方式和步数必须与编译资源一致。

| 适配器 | 已支持硬件 | 上游文件 |
| --- | --- | --- |
| Turbo4 v0.1 | 单/双 3080、单 4090 | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` |
| Turbo8 v1.0 768p | 单 4090 | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` |

固定修订与来源见[运行资源](../reference/runtime-assets#versions)。只支持表中 Ref2VA 文件；ComfyUI、FL2VA、任意自定义 LoRA 和未来上游版本需要单独适配。

Turbo4 和 Turbo8 都是蒸馏配置。减少步数可以降低计算量，但不代表输出质量与 50 步基础模型相同。名称中的 `exact` 描述所用注意力路径和指定 LoRA 的执行方式，不承诺不同 GPU 上的张量完全一致。

双卡模式已执行完整轨迹，并完成一个案例的视频与音频解码对照。它们保留精确注意力，但浮点舍入不同，不承诺输出一致或质量相同。

单卡 3080 预览版已检查容量，以及同一显卡上串行加载与传输计算重叠时的结果一致性。独立参考实现对照和更广泛的质量评测仍待完成。

## 当前版本的边界 {#scope}

公开引擎还不提供实时文本或参考素材编码、VAE 解码、MP4 输出、T2VA 或首尾帧生成，以及动态 LoRA 加载。HTTP 接口一次串行执行一个任务；账号、计费和分布式 GPU 调度由接入 Vflash 的应用负责。
