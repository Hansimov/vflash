# Veda注意力（显式SM89选项）

`attention_backend="veda-sm89"`为单SM89、原始LightX v0.1四步关键帧模型启用学习型视频稀疏注意力，
也接受显式FL/Ref `HybridModel`。这些模型的`auto`仍为dense，其他模型及双卡不接受此组合。

安装pipeline依赖后执行`python -m vflash.install_veda`，取得固定Veda修订
`fd59c7277ccc37ebf1a8f6823474b8c2ef2e33a0`的独立核心，保留MIT及SageAttention 1.0.6 BSD许可。
不安装ComfyUI或自动下载权重。[H3预测器](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview)
另行按MiniMax H3社区许可提供，实测修订为`76f202874608115408d73280be9531c4ab888242`。

Python向`H3Pipeline`传入该后端和`veda_predictor=Path("models/veda-predictor.safetensors")`。
`generate`/`denoise`使用`--attention-backend veda-sm89 --veda-predictor FILE`；原生HTTP设置
`VFLASH_ATTENTION_BACKEND=veda-sm89`及`VFLASH_VEDA_PREDICTOR`。依赖版本、权重或GPU自检失败即停止，
没有静默dense fallback。预测器由会话持有，请求间重置几何和统计，关闭会话时释放。

0–4/45–49层保持Torch Flash，其余40层实用`triton-int8`、生成保留预算0.1。
条件/文本/音频连接保持密集，但这些层的INT8算术仍近似，不能宣称保护行数值精确。
四步每请求应有160次稀疏/40次密集调用；回执记录实际后端、网格、训练计划匹配及稀疏层工作量。
8NFE训练的预测器不等于四步组合与dense数值等价。

探索性同卡同实际噪声控制：单RTX 4090 48 GB，1536×640、五秒、四步，BF16主干及运行时LoRA、
两槽block-ring、Torch2.11/CUDA13.0，冻结干净条件、共用官方解码。
Hybrid去噪102.928→72.867秒；去噪加媒体124.408→94.063秒。另一卡同期运行，
不是独占全请求或同质量保证。八时点及原尺寸脸/衣料未见新增大范围崩坏，运动改变，共有柔化尚在；
未审听音频语义。独立完整pipeline资格验证单独记录。

另完成六条实际Python全pipeline请求，重新执行官方编码和解码：两种来源，I2/L2配对、
一个较大I2画布及单图hybrid Ref。同卡896×512/五秒，I2去噪37.507→34.196秒，
L2去噪36.582→34.372秒，配对热态L2全请求64.733→61.608秒（减少4.83%）。
冷态I2总计165.293/171.789秒，分别包含94.647/101.967秒初始化及首次使用成本，不能宣称冷启动变快。
Veda I2 1344×768热态118.634秒（去噪74.010），单图Ref热态59.002秒。
不同画布不能共用一个加速倍率。大画布原尺寸脸部细节更多但成本增加，原有物体接触错误仍在；
注意力改变表情和反射。音频语义等价及普遍质量改善未建立。


## SM120显式密集层加速

在已资格的单卡RTX PRO 6000或RTX 5090上，保留`attention_backend="veda-triton"`，
为`H3Pipeline`显式指定`veda_dense_backend="sageattention2"`；完整`generate`命令支持
`--veda-dense-backend sageattention2`。默认密集层仍为Torch Flash，扩展缺失或内核失败直接停止。

需独立安装针对实际Torch/CUDA和SM120编译的
[SageAttention 2.2.0](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5)。
本次使用上述固定源码、Torch 2.11/CUDA 13.0；扩展遵循其Apache-2.0许可，本包不附带扩展或模型。
SM90专用wheel不能冒充SM120构建，CPU导入成功也不等于实际GPU资格。

仅0–4及45–49层改用INT8 QK / FP8 PV、per-warp量化、FP32+FP16累积、key smoothing，
不启用value smoothing。其余四十层Veda、保留预算、四步采样、条件和解码不变；回执记录实际
后端、密集层精度和调用数。这是显式近似算法，不承诺数值等价。

同一PRO 6000 Server、BF16 hybrid/block-ring的完整暖态I2VA对照：1536×864十五秒
Flash/Sage为257.718/223.718秒（减少13.19%），2048²五秒314.753/302.123秒（减少4.01%）；
均含条件、去噪、成片，但不含初始化和传输。1344×768五秒Flash/Sage/Flash为
49.683/49.775/52.914秒，未建立稳定短片收益。本批初始化1205.634秒，暖态收益不能代替冷启动或总费用收益。

八片全部AV解码和应用播放通过，每片五时间点及原像素细节裁剪未见新增明显结构失败；主体和建筑
结构保持，但动作轨迹改变。既有镜头遵循问题保留，音频语义未评估，不能据此宣称普遍等质。
H100和其他架构在独立资格前继续使用Flash密集层。

同一RTX 5090的1344×768五秒Flash/Sage/Flash为75.632/70.395/75.428秒；
1536×864十五秒为379.763/310.595秒（减少18.21%），修正性能分析生命周期后两者
去噪分配峰值均24.143GiB。长片已完整AV解码和应用播放，多时刻面部/服装/运动检查
保留上述已知质量限制。

同批5090另完成2048²五秒Sage368.807秒，去噪峰值26.982GiB，完整AV解码及应用播放通过；缺少本机Flash对照，因此仅确认容量和交付，不宣称4MP加速。
