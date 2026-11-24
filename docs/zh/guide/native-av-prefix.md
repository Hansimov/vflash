# 使用原生音视频潜变量续接

通过 Python 在媒体编码前保留模型输出，并为下一段请求提供兼容的历史。相邻窗口的重叠长度、源片切点和音频时钟必须一致。

## 保留原生音视频潜变量

Python 调用方可向 `H3Pipeline.generate` 传入 `latent_output=Path("clip.safetensors")`。
目标必须与 MP4 不同且尚不存在。完整的原生音视频潜变量在媒体解码前原子保存；后续解码或编码失败时仍保留。
成功返回的 `VideoResult` 包含该路径。省略此参数时维持原有临时文件生命周期。
保留文件本身不等于自动重试解码，也不代表某种续接方案已取得质量资格。

## 实验性原生音画目标前缀

`H3Pipeline.generate(..., native_av_prefix=prefix, latent_output=path)` 支持高级接口
`vflash.native.h3_av_continuation.NativeAVPrefix`。使用 `from_latents` 构建，传入原生
`video_latents`／`audio_latents`、选定的 `source_frames`、`context_frames`，以及从同一组模型权重
编译的51个 timestep 1.0 调制张量（50个主块和最终层）。这些资产由调用方显式持有，
引擎不会自行下载或换用检查点。当前仅为Python接口，CLI／HTTP准入保持原范围。

当前基础执行范围是单SM89、Veda block-ring、四步I2VA／FL2VA及一个／两个同画布关键帧；本组合拒绝重绘及性能插桩。
可选的原生声音精修必须显式保留匹配的历史，见下文。源切点和上下文均须满足24fps下 `17n+5`；上下文还须能被3整除，
例如39或90帧，以匹配40Hz音频的精确时长。源切点支持107、124等全部VAE对齐边界；
音频取最近tick，与视频切点最多相差1/120秒。首图对应上下文起点，不是其末帧。上下文占生成目标开头；
应在潜变量层去除重复区间，再对拼接后的序列统一解码。返回的五秒MP4包含上下文，不应整段追加。

`prefix.join(source_latents, generated_latents, generated_frames=124)` 核对实际前缀逐位保留，
去除重叠并返回拼接潜变量。音频长度每次按完整序列时钟重算，最多裁掉一个多余尾tick；音频不足则报错，
避免连续续接累积时钟误差。调用方随资产保留新的帧数 `source_frames + generated_frames - context_frames`。

每次Euler更新后恢复视频0.999条件强度与音频1.0干净上下文，最后恢复逐位相同的干净音画前缀。
调制表和行时刻表由当前请求持有，并在退出时恢复。960×544、五秒的实际SM89完整执行中，视频及
音频潜变量均与已实测的临时原型逐位一致；这证明接口集成保持结果，人物运动及主观音质仍逐片评价。

主动换景时，可用 `preserve_video=False` 构建前缀。每步仍恢复原生音频历史，所有目标画面行
则依据新场景首帧生成；当前同画布和执行范围约束继续适用。将目标开头 `context_frames` 帧
作为预卷去除，用 `prefix.join_audio(source_latents, generated_latents, generated_frames=124)`
取得拼接音频张量并统一解码。画面按同一帧时钟独立剪接；`join` 拒绝此模式，以免将主动换景
误作视频连续性。回执标明保留视频行数为0，`video_prefix_exact=None`。两个实际SM89换景
试验均在39帧预卷下精确保留65个历史音频tick，验证了执行与时钟；配乐连贯、对白及同步仍须试听。

若实际交付在模型补齐尾部之前结束，可显式传入 `join_audio(..., delivery_frames=N)`。
两个独立时长的舍入可能让未采用的拼接尾部少一个音频tick；该参数核对整个实际交付区间都有声音，
并保留已有原生尾部供解码，调用方随后按声明时钟裁切PCM，不补造缺失声音。
省略参数仍要求完整模型拼接时钟，包括模型补齐尾部。

可用 `vflash.compiler.continuation.compile_native_history_tables` 准备这些表。
传入本地基础transformer检查点；混合模型另传本地参考检查点和实际 `reference_blocks` 元组，
例如 `tuple(range(25, 50))`。函数仅读取对应张量，在CPU编译51个干净上下文表，返回的tuple
可直接用于 `NativeAVPrefix.from_latents`，并按模型选择复用。它不修改权重或全局线程数，
可选 `progress(completed, total)` 回调可执行调用方期限。原生前缀接口是0.22.0已包含的实验性Python接口；
0.21.0及更早wheel不包含这些接口。


续接后精修声音时，用 `compile_refinement_tables(..., include_native_history=True)` 同时编译
精修表和对应历史表。显式 `reference_directory=None` 选择全部50层原始FL；传入参考检查点则保留
现有FL前25层／Ref后25层配方，`source`记录该选择。`mode="audio"`本身不代表纯FL。
通过 `vflash.native.h3_refinement_schedule.load_refinement_history_tables(directory)` 读取配套表，
沿同一源区间构建 `preserve_video=False` 的 `NativeAVPrefix`，传给 `AudioRefinementRequest.native_audio_prefix`。
禁止把纯FL历史表与混合精修表搭配；加载器核对伴随资产声明的来源、清单、形状与有限BF16数值。
完整流水线在加载前检查主生成与历史的源区间及声音相等。已保留的输出可经原生session传入
`EncodedAudioSource` 和精修请求，无须重跑主生成或画面解码。

每次精修都在timestep1.0读取并恢复原生声音前缀，整段视频冻结，只更新随后音频。
先扩展调制存储，再构建可选权重缓存；退出时先释放缓存，再恢复原表。本路径串行执行，关闭画面解码重叠。
异常恢复原音频、视频、采样计划及调制资源。最初两个SM89九／十秒目标实测保留了65个历史tick、
全部视频潜变量和十／十八秒交付的原画面码流，但随后资产核对发现该试验将混合精修表与纯FL历史表
错误搭配；保留为失败配方，不能作为匹配音频执行的资格证明。配套编译明确防止该来源选择错误。
随后四次SM89实测在两个保留目标上比较匹配的混合／原始FL十二步配方，均保留65个历史tick和视频位。
每次采用0.25MP空间声音上下文、20GiB预算，实际缓存27个BF16块，覆盖了该组合执行；
未由此声称GPU缓存逐位等价或主观音质通过。六／二十步历史精修目前只有CPU所有权检查。


FL2VA可同时给定目标末帧与已有目标历史。两个条件关键帧均位于目标行之前，历史保留按完整条件行数偏移，
不会覆盖任何一张条件图。一次实际SM89十五秒FL2VA目标精确保留39帧视频及65个音频tick，
与源拼接为396帧。该试验的末帧约束引导了最终姿态与取景，但重复动作及中段长停留仍存在，
不能把末态到位等同自然动作或良好节奏。带历史的声音精修当前继续只接受I2VA。


`join`也接受相同的`delivery_frames`参数。若最后一帧的交付区间超过最近的40Hz刻度，
拼接保留实际已有、覆盖该帧的音频tick。例如124+124−39共209帧，完整交付需349tick，
最近时钟虽为348，也不能在确认音频充足后把第349个真实tick再裁掉。没有需要的真实音频仍报错；
默认按最近时钟拼接的行为保持。
