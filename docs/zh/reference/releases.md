# 升级至 Vflash 0.28

## 从 0.27 升级后可以做什么

- **编辑实拍源片。** [Remake 客户端](../guide/remake)分别支持编辑和替换模型。源片、顺序参考图、指定时间点的目标帧和已生成的历史片段各有独立作用。保留提交日志，连接中断后可找回原任务。
- **延长参考图视频。** [Pika 内容参考配方](../guide/pika-reference-weights)支持 5、8、12 秒，默认 5 秒。8 秒与 12 秒均在单张 96 GiB RTX PRO 6000 上完成了 1280 × 720 音视频实测。
- **校验交付媒体。** 对推理窗口，Remake 会逐帧核对 24 fps 显示时间戳，并检查方形像素和保留原声时所需的音轨。同一张图同时用作参考图与指定帧时，两种用途各自保存潜变量。

## 升级现有环境

在新环境中安装 [0.28.0 的 wheel](https://github.com/Hansimov/vflash/releases/tag/v0.28.0)，运行 `vflash doctor`。保留已固定的模型版本、源素材和提交凭据。使用 Remake 还需单独配置执行器。容器镜像通过[独立清单](https://github.com/Hansimov/vflash/blob/main/docker/images.json)发布。

[查看此前的正式发布](https://github.com/Hansimov/vflash/releases)
