# 开始使用

安装 Vflash 后，可以检查显卡并选择可用配置。实际推理还需要匹配的模型运行资源和预编译条件包。

::: info 开始前请确认
当前 alpha 版本完成的是 **预编译条件包 → 视频和音频潜变量（latents，即解码前的张量）**。运行资源包尚未公开。当前接口还不支持处理提示词、上传参考素材或输出 MP4。
:::

## 安装命令行工具 {#install}

需要 Python 3.11 或更新版本。基础安装很轻量，不会下载模型权重或 PyTorch。

```bash
git clone https://github.com/Hansimov/vflash.git
cd vflash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 检查显卡 {#check-your-gpu}

```bash
vflash doctor
vflash profiles
```

`doctor` 列出 `nvidia-smi` 能看到的 NVIDIA 显卡；`profiles` 列出当前版本支持的运行配置。

使用 `doctor` 显示的 GPU 索引，检查配置与显卡是否匹配：

```bash
# 配备 48 GB 显存的 RTX 4090
vflash plan ref2va-turbo4-exact-sm89 --gpu 0

# 配备 20 GB 显存的 RTX 3080
vflash plan ref2va-turbo4-exact-sm86 --gpu 0
```

`plan` 只检查硬件匹配情况，不加载模型权重。对于 3080 配置，已测负载的部署建议是**每个 worker 至少预留 64 GiB 可用系统内存**，并为其他进程保留额外余量。更大输入需要重新检查容量。这里的支持范围只覆盖上述显存容量；常见的 10/12 GB 版 3080 和 24 GB 版 4090 不在本次支持范围内。

完整列表见[配置与硬件](./profiles)。

## 运行条件包 {#run-a-bundle}

建议使用 [Docker](./docker) 获得固定版本的 GPU 运行环境。如果直接从 Python 源码运行，请安装 GPU 依赖：

```bash
python -m pip install -e '.[gpu]'
```

源码运行环境使用 Linux、PyTorch 2.11、Triton 3.6，以及兼容的 NVIDIA 驱动。Docker 构建固定使用 CUDA 13.0 版 PyTorch 和配套依赖。

准备相互匹配的[权重包、调度包、辅助张量文件和条件包](../reference/runtime-assets)，将下面的示例路径换成实际文件路径：

```bash
vflash denoise ref2va-turbo4-exact-sm89 \
  --gpu 0 \
  --artifact /path/to/artifact \
  --schedule-overlay /path/to/schedule \
  --auxiliary-tensor /path/to/auxiliary.safetensors \
  --bundle /path/to/example-bundle \
  --output-latents ./result.safetensors
```

使用 3080 时，选择 `ref2va-turbo4-exact-sm86`，并提供为该配置编译的资源。修改配置名称或 GPU 后缀不会自动转换资源文件。

命令会把视频和音频 latent 张量写入 `result.safetensors`，并打印 JSON 摘要。这些张量需要交给兼容的解码器；该文件还不是可播放的视频。

## 用两张 3080 执行一个请求 {#parallel}

选择两张 RTX 3080 20 GB，继续使用相同的 SM86 Turbo4 资源：

```bash
vflash plan ref2va-turbo4-exact-sm86 --gpu 0 --peer-gpu 1 --strategy tensor
```

在 `vflash denoise` 中使用相同的 `--gpu`、`--peer-gpu` 和 `--strategy` 参数即可。两张卡共同执行一个请求；这不会创建两个独立 worker。

| 策略 | 分片对象 | 选中第二张卡时的默认值 |
| --- | --- | --- |
| `tensor` | QKV、注意力输出、FFN 和 LoRA 投影权重 | 否 |
| `sequence-head` | GEMM 按 token 行分片，注意力按头分片并处理完整序列 | 是 |

`sequence-head` 从同一份主机权重向两张卡传输完整权重，用四组注意力头重叠 NCCL 通信和计算。`tensor` 传输各自的一半权重，并归约投影结果，包含原生 LoRA 分支。两种方式都使用 BF16 权重和精确注意力，执行完整四步调度，无需分布式启动器或 LightX2V 运行时。

并行计算改变 GEMM 形状或归约顺序，**不保证与单卡逐位一致**。一个负载已通过完整 latent 和解码媒体烟测，跨案例指令遵循质量仍待评测。测量边界与内存口径见[性能测量](../reference/performance#parallel)。

## 复用已加载的模型 {#reuse}

每次执行 `denoise` 都会新建会话。要连续处理多个请求，请使用 [HTTP 服务](./docker)：首个任务加载指定模型，后续任务复用已加载的权重。

## 检查失败时 {#troubleshooting}

| 现象 | 检查方法 |
| --- | --- |
| 没有列出显卡 | 运行 `nvidia-smi`，检查驱动和设备访问权限。使用 Docker 时，确认选中的 GPU 在容器内可见。 |
| 配置与显卡不匹配 | 根据 `doctor` 显示的显卡型号和显存容量选择配置。 |
| 资源缺失或不兼容 | 检查四类输入的模型、LoRA、调度和硬件版本。见[运行资源](../reference/runtime-assets)。 |
| 3080 进程耗尽系统内存 | 释放系统内存或减少 worker 数量。已测负载建议每个 worker 预留至少 64 GiB 可用 RAM；更大输入需要重新检查容量。 |
