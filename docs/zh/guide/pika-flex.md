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
