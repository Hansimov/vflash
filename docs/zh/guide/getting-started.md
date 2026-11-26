# 安装与环境检查

先安装命令行工具，查看当前进程能用的显卡，选定运行路径后再下载模型。

## 安装正式版 {#install}

使用 Python 3.11 或更高版本。以下命令从 GitHub Release 安装本版 wheel。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install https://github.com/Hansimov/vflash/releases/download/v0.31.0/vflash-0.31.0-py3-none-any.whl
python -c "import vflash; print(vflash.__version__)"
```

最后一条命令应输出 `0.29.0`。基础包提供 CLI，推理依赖随所选运行时安装。

用 CPU 即可先完成一次[源片准备与导出](./source-window)。该示例能检验基础包、FFmpeg、音轨和视频时序；实际模型推理的显卡选择见下节。

## 检查显卡 {#check-your-gpu}

```bash
vflash doctor
vflash profiles
```

在实际用于推理的进程或容器中查看显卡名称、计算能力和可用显存，然后对照[硬件要求](./profiles)。例如，具备相应显存的 SM89 设备可以运行 `vflash plan ref2va-turbo4-exact-sm89 --gpu 0` 检查四步参考图配置。


<span id="native-latents" aria-hidden="true"></span>

## 生成第一条视频 {#first-video}

1. 按[输入与交付物](./runtime-selection)选择路径。源视频提供时间线，外观参考图提供主体特征，首帧固定开场画面。
2. 使用固定配置 CLI 时，先[编译权重并准备资源](./compile-weights)，保留生成的校验文件和资源位置。
3. 运行[首帧生成示例](./complete-pipeline#first-frame)，检查 MP4 的画面和音轨。同一组资源需要连续出片时，可以[复用 Python 会话](./python)。
