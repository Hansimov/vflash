# 可选视频增强

增强独立于H3生成，**默认关闭**。保留原视频，增强另存；STCDiT能够减少坏纹理，也可能软化或改绘细节，
不能称为无损修复。推理核心不包含账号、计费、任务调度或自动坏帧检测。

完整入口为 `vflash restore-video`，显式传入 `--source-video`、新的 `--output`、
`--runtime-code`、`--weights`、`--caption-file` 和 `--trust-local-code`。
Python入口为 `vflash.restoration.restore_video`：解码、完整运动分段、增强并原子另存MP4，
复制原音轨packet而不重新编码音频。视频画面为单独H.264有损编码；已有目标文件不会覆盖。
该命令不自动借用GPU；由调用方分配设备并负责进度/取消与产品状态。

安装 `vflash[pipeline,restoration]`，按[英文运行说明](../../guide/restoration.md)准备固定版本的
作者代码、Wan1.3B基座、UMT5、VAE和tiny适配器，在单张已分配CUDA设备的隔离进程中使用
`vflash.adapters.stcdit_runtime.LocalStcditTiny`。必须显式 `trust_local_code=True`；不自动下载或
打包第三方代码/权重。权重页声明Apache2，作者代码包元数据含Apache标记但缺少完整根许可文件，
这些外部载荷的来源/许可应独立保留，不能称由vflash重新授权。

输入为原24fps时间轴的RGB帧、观察描述和覆盖全部帧的半开运动分段。增强固定BF16、10步、CFG1、
shift5、原尺寸和PyTorch注意力，不影响H3的Sol默认。校验帧数、尺寸和分段；调用前复制原图，
不改原音轨。返回增强帧，由产品以独立结果保存。已加载兼容管线可用 `StcditTinyRestorer` 直接适配。

两个17帧864²区间实测恢复/保存约127–130秒，另计加载与描述。设备采样最大约11.55GiB。
都来自同一影片：网格减少，但快动作仍虚、装饰细节变化。适配器同输入17帧与直接运行RGB完全相同，
只证明适配层一致，不代表全片根治。240帧/1,048,576像素是输入上限，不是长片或SM86资格声明；
完整播放、其它尺寸和生产接入仍需分别验证。
