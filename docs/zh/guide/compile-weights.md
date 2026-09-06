# 从官方权重准备 Ref4

此开发预览版可以从固定版本的 H3 官方权重和 Ref4 v0.1 LoRA，直接编译 BF16 Ref2VA Turbo4 运行资产。不需要捕获请求、参考图片或其他项目的实验资产。首个目标为 SM89。CPU 检查已完成；新编译器还需独立 GPU 对比后才能发布。

## 下载源文件

在此分支执行 `python -m pip install '.[pipeline]'`。编译使用 Linux、PyTorch 2.11.0、CUDA 13.0 和明确分配的 SM89 显卡。下载与文件核验仅使用 CPU。官方文件与 LoRA 合计约 146 GiB，编译输出还需要至少 48 GiB 可用磁盘空间。模型文件遵循各自的[上游许可证](../reference/license)。

以下代码按固定版本和 Vflash 内置清单下载 Ref 模型、文本及参考图编码器、官方解码器，不下载独立的 Base 模型：

```python
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from vflash.model_assets import MODEL_REVISION, upstream_inventory
from vflash.native.h3_distilled_lora import LIGHTX_H3_REF_TURBO4_CONTRACT

root = Path("models").resolve()
snapshot_download(
    "MiniMaxAI/MiniMax-H3",
    revision=MODEL_REVISION,
    allow_patterns=list(upstream_inventory()),
    local_dir=root / "minimax-h3",
)
adapter = LIGHTX_H3_REF_TURBO4_CONTRACT
hf_hub_download(
    adapter.repository,
    adapter.filename,
    revision=adapter.revision,
    local_dir=root / "adapters",
)
```

已有文件只要与固定来源逐字节一致，就可以复用。下载步骤不会执行模型仓库中的 Python 代码。后续完整链路准备步骤会核验解码器源码；实际加载仍需显式指定 `trust_local_code=True`。

## 核验与编译

先把原始权重放到最终位置，再生成本地核验记录：

```bash
python -m vflash.compiler prepare \
  --transformer models/minimax-h3/transformer_ref \
  --adapter models/adapters/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --receipt raw-weights.json
```

该命令计算全部 14 个 transformer 分片、索引、配置和完整 LoRA 文件的哈希，并核对所需 615 个基础张量、600 个 transformer LoRA 张量的形状和精度。之后执行 `python -m vflash.compiler check --receipt raw-weights.json`，仅检查文件身份和张量头，不初始化 CUDA，也不重新读取全部权重。移动或修改文件会使记录失效。

选择当前进程独占的空闲 SM89 显卡。示例中的 `0` 是此环境中 `nvidia-smi` 显示的设备编号：

```bash
python -m vflash.compiler compile \
  --receipt raw-weights.json \
  --gpu 0 \
  --output models/ref4-native
```

编译器在 CPU 上保留 BF16 大矩阵和独立 LoRA 残差。CUDA 只计算 FP32 时间嵌入 MLP，并依次处理各层 BF16 AdaLN 投影。每次计算保留原始的不同时间步数量，完成矩阵运算后才补齐表格，避免用 CPU 近似运算替代固定 CUDA 数学路径。

输出目录必须尚不存在。文件先写入同级临时目录，全部成功后才以原子操作发布，并且不会覆盖并发创建的目录。失败或取消会移除临时目录。应在独立进程运行编译，使 CUDA 上下文随命令结束而释放。

## 接入完整链路

输出包含 `artifact/`、`schedule/`、`auxiliary.safetensors` 和 `compiled.json`。资产包含全部 50 个 transformer 层；调度为四次计算，视频 shift 12、音频 shift 3；辅助包包含原生输入、输出、归一化所需的九个张量。

为[完整视频链路](./complete-pipeline)生成明确指定六个路径的配置：

```python
import json
from pathlib import Path
from vflash.pipeline import PipelineAssets, prepare_pipeline_assets

root = Path("models").resolve()
assets = PipelineAssets(
    model_directory=root / "minimax-h3",
    adapter_path=root / "adapters/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
    decoder_directory=root / "minimax-h3/FL2VA",
    artifact=root / "ref4-native/artifact",
    schedule_overlay=root / "ref4-native/schedule",
    auxiliary_tensor=root / "ref4-native/auxiliary.safetensors",
)
Path("pipeline-assets.json").write_text(json.dumps(assets.to_mapping(), indent=2))
prepare_pipeline_assets(assets, Path("prepared-assets.json"))
```

原始权重核验记录与完整链路记录用途不同：前者核验编译输入，后者核验完整生成所需的所有文件，包括编码器和解码器。两者均绑定到当前文件系统，不能在移动文件后继续使用。

新资产采用 schema 5，调度采用 schema 2。来源分别绑定基础权重摘要，以及 LoRA 的仓库、版本、摘要、alpha 8、rank 128、strength 1 和编译方法，不编造请求或 replay 标识。保留的组合 transformer 摘要与 `oracle` 字段用于匹配现有条件编码合同，不代表编译器执行过一次参考生成。旧 schema-4 资产和 schema-1 调度仍按原有来源合同读取。

此编译器不量化权重、不合并 LoRA，也不编译 Turbo8 或 Base T2VA。它不替代质量评估。更换模型、LoRA 或调度后，需要独立实现与验证。
