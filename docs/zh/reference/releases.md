# 发布说明

当前版本为 **0.2.0 正式版**。单张 RTX 4090 48 GB 支持纯文字生成视频，也支持一至三张有序参考图。完整成片和原生 latent 接口的支持范围不同，详见[支持列表](../guide/profiles)。

## 0.2.0 · 完整文生视频 {#v0-2-0}

[源码标签](https://github.com/Hansimov/vflash/tree/v0.2.0)

Python 与容器链路现在可以不传参考图，直接生成五秒、24 fps MP4。选择 `t2va-turbo4-exact-sm89`，准备固定 Base4 v1.0 资产，并省略 `--reference`。Ref4 继续支持一至三张参考图。每种模式分别持有准备记录和常驻实例，模式不匹配的请求在执行前拒绝。[模型配置指南](./pipeline-profiles)提供完整 T2VA 命令。

官方权重编译器新增 Base4 支持。在 SM89 上，全部 50 个 Base4 层、4 个调度张量和 9 个辅助张量与独立准备的 BF16 资产精确一致。实际安装包容器随后对一个固定的 928 × 512 案例执行两次完整请求：14 个条件张量和最终 FP32 音视频 latent 均与独立对照一致；重复请求的全部原始视频帧一致，两份 MP4 均完整解码，包含 120 帧和五秒双声道音轨。另一请求在完成第一次去噪后取消，移除临时输出并关闭实例。

这些检查证明该负载的实现和资源生命周期，不代表广泛的指令遵循或音频质量。Ref4 沿用 0.1.0 的多参考图验证；官方音频 VAE 重复调用仍可能出现微小浮点差异。两种完整配置均使用 BF16、固定 LoRA 和原始音频交付，不包含 W8、动态 LoRA 或首尾帧生成。SM86 完整生成尚未完成资格化，会明确拒绝；其原生接口继续可用。

镜像为任意用户 ID 提供可写的 Inductor 缓存。最终镜像核查安装包身份、支持配置与 CPU 启动；缓存目录和支持名单的调整保持已通过 GPU 验证的数值实现不变。

已发布 Linux AMD64 镜像：[`hansimov/vflash:0.2.0-pipeline`](https://hub.docker.com/r/hansimov/vflash/tags) 用于完整生成，`hansimov/vflash:0.2.0` 用于原生 HTTP 服务。两者均可匿名拉取，[不可变摘要清单](https://github.com/Hansimov/vflash/blob/v0.2.0/docker/images.json)记录源码与验证范围。模型权重按各自许可证单独下载。

## 0.1.0 · 多参考图生成视频 {#v0-1-0}

[源码标签](https://github.com/Hansimov/vflash/tree/v0.1.0)

在单张 RTX 4090 48 GB 上，通过 `vflash generate` 或 `H3Pipeline`，将提示词和一至三张图片生成五秒、24 fps MP4。图片按传入顺序对应 `<Picture N>`，原有单图 Python 参数保持兼容。`vflash prepare-pipeline` 在使用前一次性核验模型资产。[容器指南](../guide/docker#pipeline)提供完整的准备与生成命令。

实际安装包镜像在 928 × 512 下连续完成单图、双图、三图请求。三例的 14 个条件张量和最终 FP32 音视频 latent 均与固定对照逐位一致；第一例的全部原始 FP32 视频帧、第二例的 120 帧交付 RGB 也一致。三份 MP4 均完成全部视频帧与音轨解码，包含 120 帧及五秒双声道音频。三图请求在完成第一次去噪后取消，正确关闭实例、移除临时输出并释放资源。

这些是实现与生命周期验证，不代表广泛的指令质量或速度保证。原生去噪、LoRA 数学及官方媒体解码沿用 a7。官方音频 VAE 的重复调用可能出现微小浮点差异，不承诺音频逐位复现。参考图也不构成严格的关键帧时序约束。

完整镜像面向 SM89 Ref4；Ref8、T2VA Turbo4、单卡与双卡 SM86 保留原生条件包到 latent 的接口。W8、动态 LoRA 和 FL2VA 不在此次发布范围。模型权重按各自许可证单独下载。

已发布 Linux AMD64 镜像：[`hansimov/vflash:0.1.0-pipeline`](https://hub.docker.com/r/hansimov/vflash/tags) 用于完整生成，`hansimov/vflash:0.1.0` 用于原生 HTTP 服务。两者均可匿名拉取，具体镜像见[不可变摘要清单](https://github.com/Hansimov/vflash/blob/v0.1.0/docker/images.json)。

## 0.1.0a7 · 提示词与图片生成 MP4 {#a7}

[源码标签](https://github.com/Hansimov/vflash/tree/v0.1.0a7)

单张 RTX 4090 48 GB 可以从提示词和一张图片生成五秒、24 fps 的 MP4。Python 链路管理原生去噪与各阶段生命周期，并显式接入固定版本的官方文本、参考图和 VAE 适配层；支持连续请求复用、进度与取消。[开始生成视频](../guide/complete-pipeline)。

[Ref4 编译器](../guide/compile-weights) 从固定官方 H3 权重与 Turbo4 v0.1 创建运行资源，无需捕获请求，也不依赖另一个项目。基础权重与 LoRA 身份分别绑定。全部 1,250 个层内张量、4 个调度张量和 9 个辅助张量与已核验 BF16 资源精确一致。

完整链路核查得到精确一致的条件与原生 latent，解码画面也一致。官方音频 VAE 的细小浮点变化仍在排查，本版不承诺音频逐位复现或广泛的生成质量。请求总耗时包含输入准备和清理，并单列权重搬运及阶段调用。

完整链路与编译器首先支持 SM89 Ref4。T2VA Turbo4、Ref8 和 SM86 保留条件包到 latent 的接口；动态 LoRA、FL2VA 和 W8 尚未发布。Docker HTTP 服务也仍使用 latent 接口。模型权重按各自许可证另行下载。

## 0.1.0a6 · 文生视频 {#t2va}

[源码标签](https://github.com/Hansimov/vflash/tree/v0.1.0a6)

SM89 新增 T2VA Turbo4，使用 Base4 v1.0 资源。执行前核查任务、适配器身份与条件包，避免误用 Ref 权重处理纯文字请求。应用可以分别保留 Base 和 Ref 会话，减少逐任务更换模型的等待。

一个解码输出案例与会话重复调用已通过实现核查。[验证范围](../guide/profiles#t2va)说明具体负载和尚未完成的质量检查。Ref 配置沿用已有权重和计算方式。更新的 Base4 适配器、T2VA Turbo8 和 SM86 T2VA 不包含在本次发布中。

按[快速开始](../guide/getting-started)安装此源码标签。[Docker 指南](../guide/docker)从源码在本地构建 `vflash:0.1.0a6`；该历史版本未发布公开的预构建镜像。

## 0.1.0a5 · 权重读取与资源管理 {#a5}

[源码标签](https://github.com/Hansimov/vflash/tree/v0.1.0a5)

- 按文件成组读取张量，直接写入最终持有的 CPU 存储，减少重复解析文件头和中间复制。
- 关闭会话时释放其权重和锁页内存；即使调用方保留已关闭的 Python 对象，这些资源也会释放。
- 显卡或通信资源清理失败时如实报错。失败会话应关闭；无法确认清理完成时，应停止其 worker 进程。

本次保持模型权重、Turbo LoRA、去噪调度与 BF16 计算方式不变。更新改善的是加载和资源生命周期，不代表所有负载都有新的速度或质量结论。

需要固定源码版本时，从标签安装：

```bash
git clone --branch v0.1.0a5 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m pip install -e .
```

[Docker 指南](../guide/docker)使用源码在本地构建镜像，该历史版本未发布公开的预构建镜像。继续使用与配置匹配的运行资源；本次更新不会转换权重或条件包。

## 0.1.0a4 · 降低主机内存占用 {#a4}

[源码标签](https://github.com/Hansimov/vflash/tree/v0.1.0a4)

分块加载把张量放进共享的分段锁页内存，减少分配空余；加载失败时释放已经构建的部分资源。具体数据和数值对照范围见 [4090 内存测量](./benchmarks#sm89-host-memory)。

## 0.1.0a3 · 显式选择显存策略 {#a3}

[源码标签](https://github.com/Hansimov/vflash/tree/v0.1.0a3)

新增 4090 的显式分块加载选项，以及供应用接入的已完成步数回调。[a3 双 3080 数据](./benchmarks#sm86-parallel)继续保留原始版本和负载说明，不改写成当前版本的实测。

## 能力状态 {#availability}

公开包包含列表中的 **BF16 Ref2VA Turbo4 / Turbo8 和 SM89 T2VA Turbo4 去噪器**，以及**单 SM89 Ref4 与 T2VA Python / 容器完整链路与官方权重编译器**。编译器创建 SM89 的 Ref4 和 Base4 资源，其他原生配置仍需已有的匹配资源。容器镜像单独发布，不包含模型权重。

新增模式、适配器和硬件分别完成安装、数值与解码检查后，才会进入支持列表。
