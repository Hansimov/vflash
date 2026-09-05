# 使用 Docker 运行 Vflash

[English](README.md) · [完整 Docker 与 API 指南](https://hansimov.github.io/vflash/zh/guide/docker)

服务接收预编译条件包，返回视频和音频潜变量（latents，即解码前的张量）。它使用一张 GPU 或一组协作双卡，并在多个串行请求之间复用已加载的模型。

需要 Linux AMD64、Docker Compose v2、NVIDIA Container Toolkit、兼容 CUDA 13.0 的驱动、受支持的显卡和匹配的编译资源。资源包尚未公开；镜像不包含模型权重，也不提供从提示词到 MP4 的完整生成。

## 从源码启动

在仓库根目录执行：

```bash
cp docker/.env.example docker/.env
```

编辑 `docker/.env`，填写资源的绝对路径并选择显卡。保留 `VFLASH_IMAGE=vflash:0.1.0a4` 以构建当前源码。配置可选：

| 显卡 | `VFLASH_PROFILE_ID` |
| --- | --- |
| RTX 4090 48 GB | `ref2va-turbo4-exact-sm89` 或 `ref2va-turbo8-exact-sm89` |
| RTX 3080 20 GB | `ref2va-turbo4-exact-sm86` |

3080 需要为 SM86 编译的资源。对于已测负载，建议为**每个 worker 至少预留 64 GiB 可用系统内存**，并为其他进程保留额外余量；更大输入需要重新检查容量。

为 UID/GID `10001` 创建可写输出目录，把示例路径换成 `VFLASH_HOST_OUTPUTS` 对应的目录，再构建并启动：

```bash
sudo install -d -o 10001 -g 10001 /path/to/outputs
docker compose --env-file docker/.env -f docker/compose.yaml up -d --build --pull never
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
