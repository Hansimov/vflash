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
VFLASH_IMAGE=vflash:0.1.0a2
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
docker compose --env-file docker/.env -f docker/compose.yaml up -d --build --pull never
```

上述命令会从当前源码在本地构建 `vflash:0.1.0a2`。首次构建需要下载固定版本的运行依赖。模型资源以只读方式挂载，输出和内核缓存使用独立的可写存储。

Compose 默认只将接口绑定到 **127.0.0.1:8000**。引擎没有内置身份验证；本地使用时保留这个绑定，需要远程访问时则先接入应用的鉴权层。

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
  -f docker/compose.yaml -f docker/compose.parallel.yaml up -d --build --pull never
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
