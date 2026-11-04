# Pika H3 Flex参考执行

`PikaFlexRuntime`接入[官方Pika发布](https://github.com/Pika-Labs/Pika-H3-Flex)，保留上游分阶段
BF16执行，不改变现有hybrid或分段默认。当前GPU资格、输出质量和性能仍待实测；CPU合同检查不代表正式可用。

使用独立Python3.12环境，安装官方固定依赖和两个Diffusers补丁，再安装不含pipeline extra的vflash。
现有`vflash[pipeline]`固定另一套依赖，两个环境不能混装。源代码固定为
`4bf8ea508891a432106e70f675d287c27f02306d`，导入前核对22份Python/YAML文件；调用者明确授权本地代码。
适配器本身不下载代码或模型。用法和完整参数见[英文示例](../../guide/pika-flex.md)。

底模固定`42ed227ee7df40d41602854ae760620d6eb651fe`，Pika权重固定
`0ddf55ad7067ca1e76812253bd7b4600df87f759`。本地适配器根目录下使用`fl2va`与`ref2va`目录，
分别放原始manifest和safetensors。FL处理文字、首帧、尾帧和首尾帧，Ref处理有序内容参考。
上游执行前校验适配器发布哈希；调用者保留底模来源及转换回执，元数据明确本地覆盖路径的使用。

六次联合音画评估使用`1000,750,500,250,100,40,0`节点、视觉/声音shift12/3及前五步预测后的新噪声。
这不是六步Euler，不另加Larry声音阶段。FL条件为BF16，Ref条件为FP32，rank/alpha均128。

本地原生支持480p/768p、11种画幅桶、5–15整数秒请求及24fps。原始输出保留`17n+5`模型时钟，
5秒请求实际为124帧；产品裁剪、时长修整和显示比例必须单独记录。不宣称原生4K、任意画幅、
分离声音轨或无缝续接。已有成品不覆盖；一次GPU执行失败后关闭所属独立进程再恢复。

代码Apache-2.0；模型衍生物适用MiniMax H3 Community License。分发时保留原LICENSE、NOTICE、
作者修改声明及模型许可。本集成只保存固定身份，不包含权重。

## 可选组件复用

`reuse_components=True`保留一个原始BF16 transformer分区及一组原始FP32 VAE。
文本编码或解码前将transformer移到CPU；文本权重直接装入目标设备，避免同时在CPU构造第二个
完整模型。切换FL/Ref时释放旧trunk；同一trunk只注入／加载一次适配器，适配器文件或配置改变时
拒绝复用。六步采样、条件精度和原生音频计算保持原始实现。

此选项仍是生命周期候选：CPU检查覆盖复用、分区切换、小张量数值、失败清理和钩子恢复；
完整权重GPU显存、质量和速度仍待验证。不能与`cpu_offload=True`合用。使用独立进程，同一解释器内
所有`PikaFlexRuntime`共用非阻塞执行锁；`close()`释放组件。结果记录`component_reuse`和
`component_stats`实际加载／复用次数。原始分阶段路径已完成96GiB Blackwell PRO 6000上5秒480p
T2VA/I2VA生成；该范围不代表新复用路径或其他输入模式已通过。

可复用激活适配器支持明确的`kind="pika"`及原版manifest：校验完整312组rank128矩阵，
保留原参数名，使用alpha/rank=1，并按原PEFT加载器将存储的FP32适配器转为BF16。
不沿用LightX缺省alpha；部分安装失败时清除所有钩子。这只是加载合同，尚不代表量化六步
完整生成路径已完成资格化。

`PikaSamplingClock`使用分别准备的FL文本／关键帧时间投影表，核验原始六次时刻、FP32时间嵌入、
BF16调制和完整312项覆盖，并逐次检查实际模型调用。视觉时间表与最终声音配方独立，不能将
旧Euler时间表套到Pika重新加噪采样中。

原生固定画面声音适配器可用`condition_rows=0`表示直接文本来源：在时间投影校验前加入干净
视频的时间行，保留文本／声音时刻，空间抽样仍保留所有视频时间点；不创建首帧图片。
CPU检查覆盖行／时钟映射、原画面保持和清理，Pika／Larry组合的完整GPU音质仍待单独验收。
既有首帧声音路径和可选原版Pika运行时保持独立身份。


## 预备权重的 Pika 画面与 Larry 原生声音

可选的 `vflash.pipeline.pika_av_runtime.PikaAVRuntime` 复用 `ReferenceAVRuntime`
显式接收的原始 FL 行 FP8 组件，将分段会话替换为唯一持有模型的 Pika 会话。在原组件之外，
调用者需提供 `pika_source`、`pika_adapter`、`pika_manifest` 和 `pika_tables`；
`trust_local_code=True` 核验并导入固定版本的 Pika 采样代码。引擎不推断模型位置、租赁资源或产品默认。

```python
from vflash.pipeline import VideoRequest
from vflash.pipeline.pika_av import PikaAVRecipe
from vflash.pipeline.pika_av_runtime import PikaAVRuntime

# 路径、来源清单及文本加载器均由调用者明确准备和持有。
runtime = PikaAVRuntime(**prepared_components, **pika_components, trust_local_code=True)
try:
    result = runtime.generate(
        VideoRequest("一只陶瓷鸟展开翅膀。", width=960, height=544),
        output_directory,
        recipe=PikaAVRecipe(audio_steps=10, visual_attention="veda", overlap_decode=True),
    )
finally:
    runtime.close()
```

此路径当前支持五秒 T2VA／I2VA、生成原生声音及解码后的首帧，使用原版六步重新加噪与未合并的
Pika FL 适配器。`audio_steps=0` 保留六步联合音轨；10／12明确选择在固定画面上生成 Larry 原生音轨。
同请求复用提示／首帧编码，视觉结束后释放融合 QKV 副本，再执行声音及可选的重叠视频解码。
回执包含真实前向次数、注意力、条件复用、显存和完整调用计时。

该路径与前文原始 BF16 Pika 实现分开。行 FP8、Veda 和空间视频 KV 声音复用均为近似选择；
六步视觉加十步声音不等于完整请求只有六次前向。当前硬件证据覆盖完整 96 GiB SM120 PRO 6000
上的 960×544／832×480 五秒视频；其他可受理画布尚无广泛的质量／显存资格结论。生成计时不含
模型初始化和应用发布，也尚未达到 0.5MP 完整端到端低于25秒。此预备路径尚不支持 Ref2VA、
更长片段、去掉联合音频计算或精确端点重绘。

### 有界解码小块合批

`PikaAVRecipe(decode_tile_batch_size=4)`可在已准备的SM120运行时，将相同尺寸的空间解码
小块成组处理，当前支持五秒、最高960×544画布。默认仍为1，2和4为显式选项；原分块位置、
时间解码与重叠融合保持，请求结束或失败后恢复实例方法。

同一张PRO6000 Server 96GiB、600W上四场景试验保持生成音画潜变量及解码音频完全相同。
FP16批量解码和编码使8位像素略有差异，平均绝对差约1.3–1.8/255；已检查的多帧和局部未见
明显新增结构退步。相同输入的前后暖控制支持约30秒完整常驻调用减少0.7–1.1秒；其他场景
还包含首次使用／缓存差异。初始化、编排和上传另计，尚不代表完整交付低于25秒，也不扩大到
更大画布的资格。响应以`decode_tile_batches`记录实际分组。

### 严格旋转计算（源码，待发行）

此选项只在main源码中，未纳入v0.24.0固定发行包。

`PikaAVRecipe(strict_rotary=True)`在视觉和原生声音阶段显式启用已有的严格BF16半维旋转
内核，默认关闭。模型权重、采样和声音配方保持，成功或异常均恢复原函数；CPU、其他精度
或启用梯度时沿原函数执行。回执分别计数融合、原函数回退和所需连续布局复制。

同一450W PRO6000 Workstation上四组五秒960×544 T2VA／I2VA、Pika6＋Larry10、Veda及
原生VAE四块合批保持视觉潜变量、全部120帧RGB和解码音频PCM相同。运行状态较匹配的
阶段首条控制52.115秒、融合51.462秒；其他场景更大的差异包含条件缓存与暖机收益，不全
归于旋转融合。计时是完整引擎调用，不含采购、初始化和应用交付。原生成内容的质量局限
仍在；逐帧一致证明本次计算在已测案例中的等值性，不等于普遍的内容质量验收。

### 分阶段文本条件编码

`PikaAVRuntime(..., text_encoder_residency="staged")`保留原始CPU文本权重，仅在
输入条件编码期间上传到设备。条件张量继续留在设备，编码器在视觉和原生声音采样前
恢复为原CPU存储。权重、精度、提示条件和声音空间采样不变；默认仍为`"resident"`。

该选项要求`text_loader`返回CPU模型，并增加保留的主机内存。
`conditioning.text_residency`记录上传时间、CPU恢复时间和释放的设备字节。
大画幅显存受限时可组合`PikaAVRecipe(overlap_decode=False)`，使解码在声音之后执行，
避免与声音KV缓存同时占用显存。原画布、运行时和KV内存检查继续生效。
一次96GiB SM120实测完成五条五秒请求，包括两条1080P首帧请求，每次释放约29.18GB
文本编码器设备存储。同条件首帧控制成品逐字节一致；文字控制的解码画面一致，但冷暖
次序不一致时音轨有小差异。更广质量和生产采用仍待验收。

### 一次编码交付目标画布

`PikaAVRuntime.generate(..., video_crop=(left, top, right, bottom))`在首次编码前
按偶数像素边距裁剪已解码RGB，模型画布、条件、潜变量及音频采样保持。缺省不裁剪。
`OfficialMediaDecoder.generate_mp4`提供同一选项，`media.video_crop`记录源与输出尺寸。

部分解码器会对左侧H.264显示裁剪进行对齐或忽略。真实Chromium CPU夹具中，90像素
目标在元数据裁剪时显示为92像素；编码前裁剪正确显示90像素，AAC包保持一致。后续在96GiB SM120上完成三条五秒720P首帧请求，覆盖横竖画幅；
Chromium均以精确目标尺寸完整播放，AAC包序列与原0.23.1控制相同。抽样脸部局部没有
新增细节变化。这是交付修复资格，不代表动作质量或完整链路耗时已获得新的改善。
