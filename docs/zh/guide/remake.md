# Remake 源视频窗口

0.28.0a2 的实验性接口将原版 **Viggle Remake** H3 的 `swap`／`edit` 图编译并交给已准备、由调用方
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
git clone --branch v0.28.0a2 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m pip install '.[remake-client]'
```

此 CPU extra 不安装 GPU 框架。可选 `attention="veda-r2va"` 需要安装固定的
[Veda 节点](https://github.com/veda-sparse/Veda-on-ComfyUI/tree/67e4eb5541b648555563144401ebaba832bd73ec)和
[R2VA 预测器](https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/tree/1598b407905b7c0af85aa0c13acdfa6f01ff2982)。
生成与参考预算均采用其训练策略；默认 `full`。参考布局的选择是显式的，不继承 Pika 或
T2VA 预测器。已测试的稀疏路线会改变像素，应按原始场景和参考评估取舍。

## 准备输入

按[原版Remake流程](https://huggingface.co/Viggle/Viggle-Remake/blob/main/README.md)，先把对应源场景帧编辑为目标画面。肖像、产品与风格图在图像编辑阶段提供所需属性；单独肖像不等于准备好的替换场景。保持源构图与姿态，完成指定外观修改后，可将该帧pin到对应源索引。Vflash执行准备好的参考，不负责这一步图像生成。

调用方接收已编辑的源场景帧时，可直接复用像素，省去再次生图。先把相对所选源区间的
时刻映射到规范24fps时间线，再换算为所选窗口内的帧索引；图像适配该窗口画布，原始
绑定保留在调用方准备记录。源时间毫秒和产品素材ID属于调用方元数据，引擎消费的是
准备后的像素与帧索引。

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

## 可选的生成内源区域保留

0.28.0a2之后的开发版本接受 `source_mask=inputs / 'edit-mask.png'`。
这是同一输入目录拥有、与模型画布同尺寸的静态RGB图：黑色区域将原源latent作为采样条件，
白色区域继续编辑；池化到latent网格时，与白色相交的单元保留可编辑状态。
已有生成历史帧继续使用自己的干净历史条件。

在调用方拥有的Comfy环境安装相同Vflash版本，并在
`custom_nodes/VFlashRemake/__init__.py`注册节点：

```python
from vflash.pipeline.remake_conditioning import NODE_CLASS_MAPPINGS
```

它复用已有源参考的VAE编码，不做文字检测、目标跟踪或最终像素贴片。仍采用原H3的画面
条件时间，源重建和边界相互影响可能改变像素。每个准备窗口的遮罩须对齐其实际画布及补边，
需要修改的区域保持白色；屏幕固定区域不等于移动前景遮罩。请求与遮罩身份均进入离线恢复。

SM120 96GiB上一次完整四窗口、480×864编辑保住动态字幕及淡出，同时修改了保护区域外的
商品。时序取样及完整解码只支持该限定结果，自动分割、移动遮罩和泛用文字保真尚未建立。
六步采样仍为默认。

## 可选四步组合

0.28.0a2之后的开发版本还接受 `distillation="fl2v-v1.2-4step"`。
将固定[LightX2V纯FL2V v1.2适配器](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/1a4a150fc6707329409eb52e5d75a0e987ac1928)
放入运行服务的 `loras` 路径，文件为
`minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors`。
固定图通过 `LoraLoaderBypassModelOnly`、强度1加载，sigma为
`1.0, 0.947368, 0.857143, 0.666667, 0.0`，保留原合并task模型、源／声音条件及显式注意力选择。
不要把task＋turbo合并适配器再次叠加到已经包含同一task权重的模型。

原 `RemakeKeyframeUntil` 阈值仍为0.9，后两个去噪步骤低于此阈值；四步不意味着全程固定
目标关键帧。这是独立质量范围的近似配方，不是与六步等价的更快实现。初始完整编辑证据覆盖
四窗口竖屏商品片，字幕及手部／细节失败仍在。换人、静音、生成历史续接质量及其他源场景须
有各自实片证据再采用，CPU图组合检查不能代替这些范围的验证。

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
    references=(RemakeReference(inputs / 'edited-scene.png'),),
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

源字幕恢复或遮罩合成由调用方在源窗口拼装后处理，在最终画幅和时钟转换前使用相互对齐的
源帧与结果帧。这些工具有各自的依赖和质量范围；引擎接口本身不提供OCR或准确文字生成保证。

应用可把用户要求的去字幕／去水印合并到已有编辑：在同一批源场景目标帧中呈现去除后的
画面，并在六段提示中描述显露出的场景；已有Remake生成传播这些变化，省去额外OCR与
逐帧修补。仍需检查全片的文字重新出现、误改保留文字和时序伪影，单张干净目标图不等于
完整视频已通过。

若应用选择最终视频后处理，则在最终画幅／时钟转换后按交付坐标处理。两个开关分别选择，
保留原音频包与呈现时间戳，并保存未清理产物和处理回执。托管API的清理选择归应用所有，
不新增 `RemakeRequest` 或 `VideoRequest` 字段，也不改变固定推理核心；普通生成和精确
区域／时段处理保持各自已验证的范围。

准备较小的24fps模型输入之前，保留原素材的显示尺寸与帧率元数据。调用方在窗口拼装后
可另选最终画幅和帧率，并保留所选源音轨。最终帧时刻重采样改变的是交付时钟；模型的
动作采样与生成细节仍由其工作画布决定。

接口分开源归属、执行和完整拼装，后续原生渲染器可以复用同一窗口／参考／历史合同。
已有 HTTP 服务的原生潜变量 API 与普通完整流水线继续保持各自范围。

## 恢复已持久保存的完成结果

调用方在模型结果保存后、更新自身任务前退出时，可调用
`recover_completed_remake(request, input_directory=..., output_directory=...,
execution_directory=...)`读取原本地完成回执和历史。它核对原请求图、输入文件身份和
已保存MP4；不访问网络、不创建新模型执行，原运行服务可以已经停止。
不完整、已变化或缺少文件身份的旧回执仍须由原所有者明确核对，不自动重发。
普通`execute_remake`继续要求新执行目录。恢复时共同保留原输入、输出和执行回执。

该恢复合同从0.28.0a2提供，不改变采样，也不扩展场景、时长、硬件和质量资格。
