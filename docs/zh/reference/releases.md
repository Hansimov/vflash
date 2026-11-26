# 升级到 Vflash 0.31

## 从 0.30 升级

使用 `vflash prepare-remake-appearance` 从原始照片生成像素选区和视频模型参考图。回执记录方向校正后的坐标、输出尺寸及文件散列。[照片取景](../guide/remake#reference-view)


## 从 0.29 升级

同一 Remake 窗口可同时传入指定时刻的场景帧和仅提供外观的照片，总数最多四张。提示词沿用传入顺序的图片编号。`keyframe_until` 可设置关键帧调节的 sigma 截止值，默认值为 0.9。[混合参考示例](../guide/remake)


## 从 0.28 到 0.29

[准备源片与导出成片](../guide/source-window)新增两条 CPU 命令。可从原始 H.264 素材选择区间，按原构图对齐模型画布，并凭准备记录移除生成窗口中的补边与末帧重复。有声素材支持保留原有音画时差。

原生编辑在引用长历史片段时检查所选区间的逐帧显示时刻。对手机参考图按 EXIF 显示方向校验尺寸；每项参考使用单张静态图像。

### 0.29.0 配套工具

发行附件 `remake_window.py` 可配合原版 wheel 提取用于编辑的 PNG，核对成片音轨范围，并在需要时从同次准备的源片取回同期声。已安装 0.29.0 的环境可直接使用。[示例与命令](../guide/source-window)

## 安装正式版

从 [GitHub 0.31.0](https://github.com/Hansimov/vflash/releases/tag/v0.31.0) 安装 wheel，在新环境中运行 `vflash prepare-remake-source --help`。继续使用原有的模型快照和执行器。容器镜像摘要在[独立清单](https://github.com/Hansimov/vflash/blob/main/docker/images.json)中发布。

<span id="_0-16-1" aria-hidden="true"></span>
<span id="_0-17-0" aria-hidden="true"></span>
<span id="_0-18-0" aria-hidden="true"></span>
<span id="earlier" aria-hidden="true"></span>

[以前的版本](https://github.com/Hansimov/vflash/releases)
