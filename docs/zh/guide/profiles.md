# 配置与硬件

一个运行配置（profile）同时确定模型、LoRA 版本、步数、计算方式和显存使用策略。选择时，需要让配置、硬件和编译资源相互匹配。

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

**4090 配置**把权重保留在显存中，常驻服务可以在多个请求之间复用。

**3080 配置**将权重保存在锁页系统内存中，按块传入 GPU，并让传输与计算重叠。已测的 928 × 512、124 帧、4 步负载，进程占用约 **60 GiB 系统内存**。对此负载，建议**每个 worker 至少预留 64 GiB 可用系统内存**，并为操作系统和其他应用保留额外余量。更大输入和多 worker 并发需要重新测量容量；64 GiB 不是通用的容量保证。主机到 GPU 的传输速度、可用 RAM 和 GPU 算力都会影响运行效果。

当前支持范围覆盖上述 20 GB 和 48 GB 版本。两张 RTX 3080 20 GB 还可通过[权重 TP 或序列/注意力头分片](./getting-started#parallel)共同执行一个 Turbo4 请求。其他显存容量、其他显卡型号、更大的 GPU 组，以及任意分辨率或帧数组合仍不在支持范围内。

## Turbo LoRA 支持 {#lora}

Vflash 可以直接执行固定版本的 [LightX2V H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) LoRA，无需安装 LightX2V 推理框架。配置中的 LoRA、调度方式和步数必须与编译资源一致。


支持的上游文件分别固定如下：

- **Turbo4：单卡或双卡 3080、单卡 4090:** [`minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/83b617309219e859c1c264520eba07492d22e958/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors)，修订 `83b617309219e859c1c264520eba07492d22e958`.
- **Turbo8：单卡 4090:** [`minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors`](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/0eebcc7e79f9cb200927c80b8e7595265b770e34/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors)，修订 `0eebcc7e79f9cb200927c80b8e7595265b770e34`.

这些文件的摘要与 2026-09-05 核对的上游仓库修订 `2f015e66b37c585cea9dc4ae6f1850ea8788e742` 一致。支持范围仅包含这些 Ref2VA 文件，不代表支持当前或将来所有 LightX2V LoRA。ComfyUI 和 FL2VA 文件是不同的输入。

Turbo4 和 Turbo8 都是蒸馏配置。减少步数可以降低计算量，但不代表输出质量与 50 步基础模型相同。名称中的 `exact` 描述所用注意力路径和指定 LoRA 的执行方式，不承诺不同 GPU 上的张量完全一致。

双卡模式已执行完整轨迹，并完成一个案例的视频与音频解码对照。它们保留精确注意力，但浮点舍入不同，不承诺输出一致或质量相同。

单卡 3080 预览版已检查容量，以及同一显卡上串行加载与传输计算重叠时的结果一致性。独立参考实现对照和更广泛的质量评测仍待完成。

当前不支持任意 LoRA 文件，也不会自动兼容上游未来发布的新版本。版本匹配方式见[运行资源](../reference/runtime-assets)。

## 当前版本的边界 {#scope}

公开引擎还不提供实时文本或参考素材编码、VAE 解码、MP4 输出、T2VA 或首尾帧生成，以及动态 LoRA 加载。HTTP 接口一次串行执行一个任务；账号、计费和分布式 GPU 调度由接入 Vflash 的应用负责。
