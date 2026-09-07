# Docker 与 API

把 Vflash 作为本地 HTTP 服务运行。服务接收预编译条件包，返回视频和音频潜变量（latents，即解码前的张量）；加载一个固定配置后，连续串行处理多个请求。

## 运行要求 {#requirements}

需要 Linux AMD64、Docker Compose v2、NVIDIA Container Toolkit，以及兼容镜像内 CUDA 13.0 运行时的 NVIDIA 驱动。另外还需要一张受支持显卡或一组双 3080，见[硬件列表](./profiles)和四类[运行资源](../reference/runtime-assets)。

对于已测的 3080 负载，建议每个 worker 至少预留 **64 GiB 可用系统内存**，并为其他进程保留额外余量。更大输入需另做[容量检查](./profiles#memory)。模型资源单独挂载，不包含在镜像中。

## 配置并启动 {#start}

在 Vflash 源码目录执行：

```bash
cp docker/.env.example docker/.env
```

编辑 `docker/.env`，把所有示例路径换成 Docker 主机上的绝对路径：

```dotenv
VFLASH_IMAGE=hansimov/vflash:0.3.1
VFLASH_PROFILE_ID=ref2va-turbo4-exact-sm89
VFLASH_GPU_DEVICE=0

VFLASH_HOST_ARTIFACT=/path/to/artifact
VFLASH_HOST_SCHEDULE=/path/to/schedule
VFLASH_HOST_AUXILIARY=/path/to/auxiliary.safetensors
VFLASH_HOST_BUNDLES=/path/to/bundles
VFLASH_HOST_OUTPUTS=/path/to/outputs
```

`VFLASH_GPU_DEVICE` 使用索引或 UUID 选中一张主机显卡。容器内会把它显示为设备 `0`。使用 3080 时，将配置改为 `ref2va-turbo4-exact-sm86` 并提供匹配的 SM86 资源；使用 4090 Turbo8 时，选择 `ref2va-turbo8-exact-sm89` 及相应资源。

容器使用 UID/GID `10001` 运行。为这个用户创建可写输出目录，再从当前源码构建并启动服务：

```bash
sudo install -d -o 10001 -g 10001 /path/to/outputs
docker compose --env-file docker/.env -f docker/compose.yaml pull
docker compose --env-file docker/.env -f docker/compose.yaml up -d --no-build
```

带版本号的镜像已发布到 [Docker Hub](https://hub.docker.com/r/hansimov/vflash/tags)，不可变摘要见[镜像清单](https://github.com/Hansimov/vflash/blob/v0.3.1/docker/images.json)。模型只读挂载，输出和内核缓存使用独立的可写存储。需要本地构建时，执行 `docker build --target runtime -t vflash:0.3.1 .`，然后在环境文件中选择该镜像。

Compose 默认只将接口绑定到 **127.0.0.1:8000**。引擎没有内置身份验证；本地使用时保留这个绑定，需要远程访问时则先接入应用的鉴权层。

## 在容器中生成 MP4 {#pipeline}

`pipeline` 镜像预装官方编码器与 VAE 适配器、CUDA 对应的 Torchvision、FFmpeg 和 FFprobe，入口是 `vflash`，用 `generate` 生成完整视频。独立的原生镜像运行 HTTP latent 服务。

拉取已发布的完整链路镜像：

```bash
docker pull hansimov/vflash:0.3.1-pipeline
mkdir -p inputs outputs cache
```

将包含六个资产路径的 `pipeline-assets.json`、提示词与参考图放入 `inputs`。JSON 中的模型路径使用容器内最终的 `/models/...` 路径，并将对应的主机模型目录只读挂载。在最终挂载布局内准备记录；此步骤仅核验一次资产字节，不需要显卡：

```bash
docker run --rm --runtime=runc -e NVIDIA_VISIBLE_DEVICES=void \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,mode=1777 \
  -v /absolute/model-directory:/models:ro \
  -v "$PWD/inputs:/inputs:ro" -v "$PWD/outputs:/outputs:rw" \
  -v "$PWD/cache:/cache:rw" \
  hansimov/vflash:0.3.1-pipeline prepare-pipeline \
  --assets /inputs/pipeline-assets.json --receipt /outputs/prepared-assets.json
```

在一张 RTX 4090 48 GB 上生成。参考图的传入顺序对应提示词中的 `<Picture N>`：

```bash
docker run --rm --gpus device=0 --shm-size 4g \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,mode=1777 \
  -v /absolute/model-directory:/models:ro \
  -v "$PWD/inputs:/inputs:ro" -v "$PWD/outputs:/outputs:rw" \
  -v "$PWD/cache:/cache:rw" \
  hansimov/vflash:0.3.1-pipeline generate \
  --prepared-assets /outputs/prepared-assets.json \
  --prompt-file /inputs/prompt.txt \
  --reference /inputs/subject.png --reference /inputs/setting.png \
  --output /outputs/video.mp4 --gpu 0 --seed 1234 --trust-local-code
```

使用 Ref4 时，`--reference` 可以出现一至三次。使用 T2VA 时，以 `--profile t2va-turbo4-exact-sm89` 准备 Base4 资产，并省略参考图。两者均生成五秒、24 fps 视频。进度 JSON 写入 stderr，最终结果写入 stdout。镜像不包含模型权重；请核对[完整链路能力和内存预算](./complete-pipeline)及[模型许可证](../reference/license)。

如需从标签源码构建完整镜像，可执行 `docker build --target pipeline -t vflash:0.3.1-pipeline .`，并将命令中的镜像名替换为本地名称。

双张 RTX 3080 20 GB 使用 [SM86 编译资产](../reference/pipeline-profiles#sm86)，在 `prepare-pipeline` 后加 `--profile ref2va-turbo4-exact-sm86`。生成命令将 `--gpus device=0` 替换为 `--gpus '"device=0,1"'`，在 `--gpu 0` 后加 `--peer-gpu 1 --strategy sequence-head`。进程退出前，两张卡都由该实例持有；主卡执行编码和解码，双卡共同去噪。这里是完整链路配置，下文的 Compose 则部署原生 HTTP 接口。

已发布镜像复用不可变的 0.3.0 镜像层，安装 0.3.1 wheel。[发行附件](https://github.com/Hansimov/vflash/releases/tag/v0.3.1)包含该 wheel 和 `Dockerfile.release`；[清单](https://github.com/Hansimov/vflash/blob/v0.3.1/docker/images.json)记录其摘要、依赖镜像与验证范围。这样不必重新下载未变更的依赖；上面的源码 Dockerfile 仍提供完整构建方法。

镜像默认使用 UID/GID `10001`；示例改用当前用户，方便写入输出。Jiterator 直接使用可写的 `/cache` 根目录，也支持新挂载的空目录。显式设置 Inductor 缓存后，即使这个用户未登记在容器中也能运行；此设置也适用于 0.1.0 镜像。`/cache` 必须允许加载编译后的共享库，不要放在 `noexec` 挂载点。准备记录中的资产路径和文件身份必须保持一致。连续生成时，建议在独立容器进程中复用 Python `H3Pipeline`，避免每次执行命令都重新加载模型。

## 双卡协作 {#parallel}

使用两张 RTX 3080 20 GB 时，提供 SM86 权重与调度资源，并设置：

```dotenv
VFLASH_GPU_DEVICE=0
VFLASH_PEER_GPU_DEVICE=1
VFLASH_PARALLEL_STRATEGY=sequence-head
```

标准权重张量并行使用 `tensor`。需要 Docker Compose **2.24.4 或更新版本**，添加双卡覆盖配置：

```bash
docker compose --env-file docker/.env \
  -f docker/compose.yaml -f docker/compose.parallel.yaml up -d --no-build
```

覆盖配置选择 SM86 Turbo4，向容器提供选中的两张主机显卡，并为 NCCL 设置 1 GiB 共享内存。一个 worker 持有两张卡，串行处理请求。增加 worker 时需分配互不重叠的显卡组。此模式已在没有 peer access 的 PCIe 3.0 x16 主机桥接拓扑上测量，无需 NVLink。

该文件通过 Compose 的 [`!override` 合并规则](https://docs.docker.com/reference/compose-file/merge/#replace-value)替换显卡预留配置。就绪检查同时检查两张卡；任一 rank 失败后会关闭整个执行组，需重启服务再提交任务。

## 检查就绪状态 {#readiness}

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/v1/profiles
```

`/healthz` 检查 HTTP 进程是否存活；`/readyz` 检查配置文件、输出目录、显卡和 worker 状态。模型在首个任务中加载，因此就绪检查通过不代表模型已经驻留。

## 提交任务并下载 {#jobs}

如果条件包位于 `VFLASH_HOST_BUNDLES/example-bundle`，提交它的相对目录名：

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/denoise/jobs \
  -H 'content-type: application/json' \
  -d '{"bundle":"example-bundle"}'
```

使用返回的 `id` 查询任务。状态依次为 `queued`（排队）、`running`（运行），最终变为 `succeeded`（成功）或 `failed`（失败）。

```bash
curl -fsS http://127.0.0.1:8000/v1/denoise/jobs/JOB_ID
```

状态为 `succeeded` 后，下载 latent 输出：

```bash
curl -fLo result.safetensors \
  http://127.0.0.1:8000/v1/denoise/jobs/JOB_ID/output
```

该文件包含张量，还不是可播放的视频。任务成功前请求下载会返回 `409`。

## 队列与恢复 {#queue}

一个 CUDA worker 一次执行一个任务。可在 `docker/.env` 中设置以下选项：

| 设置 | 默认值 | 含义 |
| --- | ---: | --- |
| `VFLASH_API_PORT` | `8000` | Compose 绑定到本机回环地址的端口 |
| `VFLASH_JOB_TIMEOUT_SECONDS` | `1800` | worker 请求的最长执行时间 |
| `VFLASH_MAX_PENDING_JOBS` | `8` | 排队中与执行中任务的数量上限 |
| `VFLASH_JOB_HISTORY_LIMIT` | `128` | 内存中保留的已完成任务记录上限 |

队列满时，提交返回 `429` 和 `Retry-After`。调用方应等待指定时间后再重试。

任务记录保存在内存中，重启后会丢失，较早的已完成记录也会被移除。移除记录不会删除输出文件，但不能再通过对应 API 查询或下载。调用方应及时下载结果，并自行管理持久任务记录和文件保留策略。

CUDA worker 退出或超时后，就绪检查会失败，需要重启服务恢复。任务不会自动重放。已经完成的文件仍保留在输出挂载目录中。

计时字段和首次请求开销见[性能测量](../reference/performance)。

## API 参考 {#api}

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/healthz` | 检查 HTTP 进程 |
| `GET` | `/readyz` | 检查文件、显卡和 worker 就绪状态 |
| `GET` | `/v1/profiles` | 查看服务当前使用的配置 |
| `POST` | `/v1/denoise/jobs` | 提交条件包 |
| `GET` | `/v1/denoise/jobs/{id}` | 查询任务状态和结果元数据 |
| `GET` | `/v1/denoise/jobs/{id}/output` | 下载已完成的 latent 文件 |

启动服务后，可在 `http://127.0.0.1:8000/docs` 查看交互式 OpenAPI 文档。

## 不使用 Docker 时 {#python-server}

安装 `.[gpu,server]`，在进程环境中设置路径和配置，再启动一个服务进程：

```bash
python -m vflash.server
```

直接运行 Python 服务时，使用 `VFLASH_ARTIFACT_PATH`、`VFLASH_SCHEDULE_OVERLAY_PATH`、`VFLASH_AUXILIARY_TENSOR_PATH`、`VFLASH_BUNDLE_ROOT` 和 `VFLASH_OUTPUT_ROOT` 指定本地路径，使用 `VFLASH_GPU_INDEX` 指定物理 GPU 索引。本地访问请设置 `VFLASH_API_HOST=127.0.0.1`；Python 入口默认绑定所有网络接口。配置、超时、队列和历史记录选项使用本页中的同名变量。

双卡 Python 服务设置 `VFLASH_PEER_GPU_INDEX`，需要权重 TP 时另设 `VFLASH_PARALLEL_STRATEGY=tensor`。指定第二张卡后，默认策略为 `sequence-head`。一个 worker 独占这两张物理显卡。
