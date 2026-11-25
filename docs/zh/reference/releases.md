# 升级至 Vflash 0.29

## 从 0.28 到 0.29

[准备源片与导出成片](../guide/source-window)新增两条 CPU 命令。可从原始 H.264 素材选择区间，按原构图对齐模型画布，并凭准备记录移除生成窗口中的补边与末帧重复。有声素材支持保留原有音画时差。

原生编辑在引用长历史片段时检查所选区间的逐帧显示时刻。对手机参考图按 EXIF 显示方向校验尺寸；每项参考使用单张静态图像。

### 0.29.0 配套工具

发行附件 `remake_footage_checked.py` 可配合原版 wheel 提取用于编辑的 PNG，核对成片音轨范围，并在需要时从同次准备的源片取回同期声。已安装 0.29.0 的环境可直接使用。[示例与命令](../guide/source-window)

## 安装正式版

从 [GitHub 0.29.0](https://github.com/Hansimov/vflash/releases/tag/v0.29.0) 安装 wheel，在新环境中运行 `vflash prepare-remake-source --help`。继续使用原有的模型快照和执行器。容器镜像摘要在[独立清单](https://github.com/Hansimov/vflash/blob/main/docker/images.json)中发布。

<span id="_0-16-1" aria-hidden="true"></span>
<span id="_0-17-0" aria-hidden="true"></span>
<span id="_0-18-0" aria-hidden="true"></span>
<span id="earlier" aria-hidden="true"></span>

[以前的版本](https://github.com/Hansimov/vflash/releases)
