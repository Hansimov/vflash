# Remake 源视频窗口

0.28.0a1 的实验性接口将原版 **Viggle Remake** H3 的 `swap`／`edit` 图编译并交给已准备、由调用方
拥有的本地 Comfy 运行服务执行。请显式选择下方预览 tag；稳定版 0.27.0 wheel 不包含此接口。普通 Pika 与原生执行配置保留现有接口。

## 运行时与模型

安装沿[原作者说明](https://huggingface.co/Viggle/Viggle-Remake/blob/7e5be143702dd53b9154482de673b8519ed2d4d6/COMFYUI.md)：

| 部分 | 固定配置 |
| --- | --- |
| Remake 源码与模型 | [7e5be143](https://huggingface.co/Viggle/Viggle-Remake/tree/7e5be143702dd53b9154482de673b8519ed2d4d6) |
| ComfyUI | [355cd6b1](https://github.com/Comfy-Org/ComfyUI/tree/355cd6b14c92c19ac3188fe528e47a3f21ef5178)／0.42.0 |
| 原图实测环境 | Python 3.12.15、Torch 2.9.1+cu130、comfy-kitchen 0.2.40 |
| 画面模型 | 原作者已合并 INT8 的 swap 或 edit；分别采用其六步 sigma |
| 条件与解码 | Qwen NVFP4、原生画面和声音 VAE |

模型目录须包含 `remake_graph` 声明的文件。运行服务所有者提供原版节点包、模型路径、输入
与输出目录，负责启动、GPU 归属、就绪及停止；该独立 Python 环境与原生 `pipeline` extra
分开。适配器只提交固定图。模型许可证及 Viggle 署名要求独立于 Vflash 的源码许可证。


从固定预览 tag 安装 CPU 客户端，并提供系统 `ffmpeg`／`ffprobe`：

```bash
git clone --branch v0.28.0a1 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m pip install '.[remake-client]'
```

此 CPU extra 不安装 GPU 框架。可选 `attention="veda-r2va"` 需要安装固定的
[Veda 节点](https://github.com/veda-sparse/Veda-on-ComfyUI/tree/67e4eb5541b648555563144401ebaba832bd73ec)和
[R2VA 预测器](https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/tree/1598b407905b7c0af85aa0c13acdfa6f01ff2982)。
生成与参考预算均采用其训练策略；默认 `full`。参考布局的选择是显式的，不继承 Pika 或
T2VA 预测器。已测试的稀疏路线会改变像素，应按原始场景和参考评估取舍。

## 准备输入

`RemakeRequest` 明确声明路径与角色：

- 完整 H.264 源窗口、24fps、明确画布及 `17n + 5` 帧数（90–345）。保留声音时传入真实源音轨。
- 一到四张 `RemakeReference`。无 pin 的外观／风格参考，短边与生成画布一致，宽高均为32倍数；
  由调用方等比裁缩或补边，不通过拉伸改变人或物体的比例。
- 有 pin 的参考是指定帧的真实目标画面，尺寸与源画布完全相同；全部参考设 pin 或全部不设。
- 可选 `RemakePrefix(take, start, frames)` 指向上一条实际生成画面，历史帧数为17的倍数。
  原源视频和已生成历史是不同的输入。

固定图接收一段显式源与顺序 `<Picture N>` 参考。人物识别、目标图、切镜和整片分窗由调用方
负责。输入上界是校验范围，质量与容量仍按任务和设备确定。原图 GPU 证据覆盖 SM120 96GiB、
832×480、124／192／226帧、人物／物体／风格编辑、两个身份及一次17帧历史续接。
多次长片续接、其他画布及此新传输入口的实际部署仍待资格验证。

## 执行一个窗口

```python
from pathlib import Path
from vflash.pipeline.remake import RemakeReference, RemakeRequest
from vflash.pipeline.remake_execution import LocalRemakeRuntime, execute_remake

inputs = Path('/srv/remake/input')
request = RemakeRequest(
    source=inputs / 'source.mp4',
    prompt=Path('prompt.txt').read_text(),  # 原版H3六字段编辑提示。
    mode='swap', frames=124, width=832, height=480,
    references=(RemakeReference(inputs / 'appearance.png'),),
    soundtrack='preserve', seed=0,
)

# runtime_owner 是应用已有的资源／进程控制器。
# stop_and_confirm() 停止本次运行服务，并在确认后返回True。
result = execute_remake(
    request,
    input_directory=inputs,
    output_directory=Path('/srv/remake/output'),
    execution_directory=Path('/srv/remake/executions/window-001'),
    runtime=LocalRemakeRuntime('http://127.0.0.1:8188'),
    stop_runtime=runtime_owner.stop_and_confirm,
    timeout_seconds=600,
    cancel_check=runtime_owner.cancelled,
)
print(result['artifact'], result['media'])
```

每次使用新的私有执行目录。提交前持久保存原 UUID、完整图与状态；回包丢失后只观察同一
UUID，不重发 POST。已有目录保留给显式恢复。同一运行服务对象只接受一个执行通道，
任务串行由其所有者调度。

完成时核对实际执行图、固定 SaveVideo 产物目录，以及真实 MP4 的帧数、画布和声音。
LoadVideo 的输入预览不会被当成结果。取消、超时和不确定执行调用原所有者的停止操作；
未确认停止会保留在记录中，继续由资源所有者处理。HTTP 仅接受本机回环地址，不跟随
重定向或环境代理。

## 还原完整源时间线

返回产物是**模型窗口**。调用方去除生成历史和模型补帧，让每个真实源帧只出现一次，
裁去明确的模型边框，并确定最终声音。`preserve` 将源声音作为干净模型条件，但原图
仍经过声音 VAE 解码；要保持原声，可在最终拼装时直接复制完整规范源的 AAC。
`silent` 的原始产物不带音轨。

接口分开源归属、执行和完整拼装，后续原生渲染器可以复用同一窗口／参考／历史合同。
已有 HTTP 服务的原生潜变量 API 与普通完整流水线继续保持各自范围。
