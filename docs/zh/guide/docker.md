# 部署推理服务

原生服务持有一个模型配置，在所属设备上串行执行去噪。客户端提交挂载目录里的条件包名称，模型由服务端持有。公开 Docker 镜像有独立版本 0.21.0；需要当前 Python 包时应从本版源码构建。

## 配置挂载与设备 {#start}

在指定版本的源码目录中复制环境模板，设置镜像、配置、GPU 及宿主机绝对路径。模型输入只读挂载，输出目录给予容器用户 UID/GID 10001 写权限。先验证单卡路径。

```bash
cp docker/.env.example docker/.env
```

```bash
sudo install -d -o 10001 -g 10001 /path/to/outputs
docker compose --env-file docker/.env -f docker/compose.yaml pull
docker compose --env-file docker/.env -f docker/compose.yaml up -d --no-build
```

## 从正式标签构建 0.28.0

需要当前包的运行时时，请在 v0.28.0 源码目录内执行下列命令。镜像使用独立的本地名称，便于区分清单中的已发布镜像。在同一个终端设置 VFLASH_IMAGE 后，Compose 会启动刚校验的构建。

```bash
docker build --target runtime -t vflash-local:0.28.0 .
docker run --rm --entrypoint python vflash-local:0.28.0 -c 'import vflash; print(vflash.__version__)'
export VFLASH_IMAGE=vflash-local:0.28.0
docker compose --env-file docker/.env -f docker/compose.yaml up -d --no-build
```

## 检查运行中的服务 {#readiness}

分别检查进程健康、就绪状态及选定配置。即使进程存活，服务仍可能缺少匹配的显卡或模型资源。用小型已知条件包提交一次并下载张量，可验证完整链路。

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/v1/profiles
```

## 管理队列

提交与取回结果见 [HTTP 参考](../reference/http-api)。原生队列只在内存中保留有限历史，持久任务记录和完成文件应由调用方保存。在同一主机新增 worker 前应核对显存与可用主机内存。

## 在容器中生成 MP4 {#pipeline}

将已经准备好的模型路径写入 inputs/pipeline-assets.json，提示词与参考图置于 inputs。下面两步使用完全相同的容器内路径：先校验资源并保存凭据，再用 GPU 生成完整视频。宿主机的模型目录需替换为实际绝对路径。

```bash
docker pull hansimov/vflash:0.21.0-pipeline
mkdir -p inputs outputs cache
```

```bash
docker run --rm --runtime=runc -e NVIDIA_VISIBLE_DEVICES=void \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,mode=1777 \
  -v /absolute/model-directory:/models:ro \
  -v "$PWD/inputs:/inputs:ro" -v "$PWD/outputs:/outputs:rw" \
  -v "$PWD/cache:/cache:rw" \
  hansimov/vflash:0.21.0-pipeline prepare-pipeline \
  --assets /inputs/pipeline-assets.json --receipt /outputs/prepared-assets.json
```

```bash
docker run --rm --gpus device=0 --shm-size 4g \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,mode=1777 \
  -v /absolute/model-directory:/models:ro \
  -v "$PWD/inputs:/inputs:ro" -v "$PWD/outputs:/outputs:rw" \
  -v "$PWD/cache:/cache:rw" \
  hansimov/vflash:0.21.0-pipeline generate \
  --prepared-assets /outputs/prepared-assets.json \
  --prompt-file /inputs/prompt.txt \
  --reference /inputs/subject.png --reference /inputs/setting.png \
  --output /outputs/video.mp4 --gpu 0 --seed 1234 --trust-local-code
```

## 两张显卡协作 {#parallel}

两张受支持的 SM86 显卡可以共同执行一个配置。把两个设备写入 docker/.env，按[硬件要求](./profiles)选择对应的双卡配置，再叠加协作配置启动服务。

```dotenv
VFLASH_GPU_DEVICE=0
VFLASH_PEER_GPU_DEVICE=1
VFLASH_PARALLEL_STRATEGY=sequence-head
```

```bash
docker compose --env-file docker/.env \
  -f docker/compose.yaml -f docker/compose.parallel.yaml up -d --no-build
```
