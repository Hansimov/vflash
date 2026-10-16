# 开始使用

安装当前发行版，选择经过验证的配置并准备对应资产。`H3Pipeline`/`vflash generate`生成完整MP4；
`denoise`与原生HTTP服务接收预编译条件并返回潜变量。模型下载和编译均为显式步骤。

当前模式、时长与参考限制以[完整链路合同](./complete-pipeline)为准，GPU显存与架构资格以
[配置指南](./profiles)为准。引擎不会通过暗中更换模型、设备或精度来容纳超限请求。

## 安装命令行工具 {#install}

需要 Python 3.11 或更新版本。基础安装很轻量，不会下载模型权重或 PyTorch。

```bash
git clone --branch v0.14.0 --depth 1 https://github.com/Hansimov/vflash.git
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

# 两张配备 20 GB 显存的 RTX 3080
vflash plan ref2va-turbo4-exact-sm86 --gpu 0 --peer-gpu 1
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

## 用两张显卡执行一个请求 {#parallel}

选择两张 RTX 3080 20 GB，继续使用相同的 SM86 Turbo4 资源：

```bash
vflash plan ref2va-turbo4-exact-sm86 --gpu 0 --peer-gpu 1 --strategy sequence-head
```

在 `vflash denoise` 中使用相同的 `--gpu`、`--peer-gpu` 和 `--strategy` 参数即可。两张卡共同执行一个请求；这不会创建两个独立 worker。

| 策略 | 分片对象 | 选中第二张卡时的默认值 |
| --- | --- | --- |
| `tensor` | QKV、注意力输出、FFN 和 LoRA 投影权重 | 否 |
| `sequence-head` | GEMM 按 token 行分片，注意力按头分片并处理完整序列 | 是 |

两种方式均使用 BF16 权重、精确注意力和完整的四步调度。选择第二张卡后，默认策略是 `sequence-head`。实现说明见[双卡架构](../reference/architecture#parallel)。

0.4.0 还允许已准备的 SM89 Base16 I2VA/L2VA/FL2VA profile 使用两张匹配的 RTX 4090 48 GB。传入同样的 peer 参数，但只能选择 `sequence-head`；`tensor` 仍会被拒绝。两张卡从同一份已准备的 SM89 工件分块流入权重。这是优先降低单个请求延迟的可选路径；已有两个就绪请求时，不应取代两个独立 worker。

并行计算改变 GEMM 形状或归约顺序，因此结果**不一定与单卡逐位一致**。一个负载已通过完整 latent 与解码后视频、音频的基础检查，跨案例指令遵循质量仍待评测。测量边界与内存口径见[性能测量](../reference/performance#parallel)。

## 复用已加载的模型 {#reuse}

连续处理多个请求时，使用 [Python 会话](./python)或 [HTTP 服务](./docker)。Python 会话在创建时加载模型，HTTP 服务在首个任务加载，后续请求继续复用；每次重新运行 `vflash denoise` 则会重新加载。

<span id="python-lifetime"></span>

会话的创建、进度回调和关闭规则见 [Python 集成](./python#lifetime)。

## 检查失败时 {#troubleshooting}

| 现象 | 检查方法 |
| --- | --- |
| 没有列出显卡 | 运行 `nvidia-smi`，检查驱动和设备访问权限。使用 Docker 时，确认选中的 GPU 在容器内可见。 |
| 配置与显卡不匹配 | 根据 `doctor` 显示的显卡型号和显存容量选择配置。 |
| 资源缺失或不兼容 | 检查四类输入的模型、LoRA、调度和硬件版本。见[运行资源](../reference/runtime-assets)。 |
| 3080 进程耗尽系统内存 | 释放系统内存或减少 worker 数量。已测负载建议每个 worker 预留至少 64 GiB 可用 RAM；更大输入需要重新检查容量。 |
