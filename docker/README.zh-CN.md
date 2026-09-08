# 使用 Docker 运行 Vflash

[English](README.md) · [完整 Docker 与 API 指南](https://hansimov.github.io/vflash/zh/guide/docker)

使用 [pipeline 镜像](https://hansimov.github.io/vflash/zh/guide/docker#pipeline)，可以在 SM89 上用纯文字或提示词与一至三张参考图生成完整 MP4，也支持双 SM86 的参考图生成。下文介绍独立的原生 HTTP 服务。

服务接收预编译条件包，返回视频和音频潜变量（latents，即解码前的张量）。它使用一张 GPU 或一组协作双卡，并在多个串行请求之间复用已加载的模型。

需要 Linux AMD64、Docker Compose v2、NVIDIA Container Toolkit、兼容 CUDA 13.0 的驱动、受支持的显卡和匹配的编译资源。镜像不含模型权重，提供条件包到 latent 的接口。独立的[权重编译器](../docs/zh/guide/compile-weights.md)可以准备 SM86/SM89 的 Ref4 资产，以及 SM86/SM89 的 Base4 资产；[完整 MP4 生成](../docs/zh/guide/complete-pipeline.md)使用 pipeline 镜像或 Python 扩展。

## 启动发布版

在仓库根目录执行：

```bash
cp docker/.env.example docker/.env
```

编辑 `docker/.env`，填写资源的绝对路径并选择显卡。使用 `VFLASH_IMAGE=hansimov/vflash:0.3.2` 拉取已发布镜像。配置可选：

| 显卡 | `VFLASH_PROFILE_ID` |
| --- | --- |
| RTX 4090 48 GB | `ref2va-turbo4-exact-sm89`、`ref2va-turbo8-exact-sm89` 或 `t2va-turbo4-exact-sm89` |
| RTX 3080 20 GB | `ref2va-turbo4-exact-sm86` |
| 双 RTX 3080 20 GB | `t2va-turbo4-exact-sm86`，使用 `sequence-head` |

3080 需要为 SM86 编译的资源。对于已测负载，建议为**每个 worker 至少预留 64 GiB 可用系统内存**，并为其他进程保留额外余量；更大输入需要重新检查容量。

为 UID/GID `10001` 创建可写输出目录，把示例路径换成 `VFLASH_HOST_OUTPUTS` 对应的目录，再构建并启动：

```bash
sudo install -d -o 10001 -g 10001 /path/to/outputs
docker compose --env-file docker/.env -f docker/compose.yaml pull
docker compose --env-file docker/.env -f docker/compose.yaml up -d --no-build
curl -fsS http://127.0.0.1:8000/readyz
```

Compose 默认只绑定本机回环地址。引擎没有身份验证；远程访问需要先接入应用的鉴权层。

## 提交条件包

条件包位于 `VFLASH_HOST_BUNDLES/example-bundle` 时：

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/denoise/jobs \
  -H 'content-type: application/json' \
  -d '{"bundle":"example-bundle"}'

curl -fsS http://127.0.0.1:8000/v1/denoise/jobs/JOB_ID
```

状态为 `succeeded` 后下载输出：

```bash
curl -fLo result.safetensors \
  http://127.0.0.1:8000/v1/denoise/jobs/JOB_ID/output
```

队列有容量上限，超载时返回 `429` 和 `Retry-After`。任务记录是临时的，重启或历史记录淘汰后会消失，但输出文件仍留在磁盘。调用方应及时下载结果，并自行管理持久任务记录和输出保留策略。

就绪检查、配置项、超时恢复和全部 API 见[完整指南](https://hansimov.github.io/vflash/zh/guide/docker)。服务运行时，可在 `http://127.0.0.1:8000/docs` 查看交互式 OpenAPI 文档。

使用两张 RTX 3080 20 GB 时，加入 `docker/compose.parallel.yaml` 并设置 `VFLASH_PEER_GPU_DEVICE`。`tensor` 与 `sequence-head` 的配置方式见[双卡部署](https://hansimov.github.io/vflash/zh/guide/docker#parallel)。
