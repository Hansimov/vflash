# 选择注意力后端

注意力后端与模型配方、GPU 架构相关。先按配置默认值跑通一条完整视频，再固定输入比较替代后端。稀疏层布局会改变画面。

## 在 SM89 上运行关键帧生成

单张 SM89 显卡运行原版四步 LightX 关键帧配置或显式 FL/Ref 混合模型时，可选择 `veda-sm89`。执行 `python -m vflash.install_veda` 安装固定版本的扩展，单独取得 [H3 预测器](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview)。启动时同时传入 `--attention-backend veda-sm89` 和 `--veda-predictor FILE`。依赖版本、预测器或设备自检失败时，请求会停止并报告原因。

## 验证部署配方

在执行记录中保留稀疏层和稠密层分配、稠密内核、精度和模型。Hopper 与 Blackwell 需要各自的内核与实测。固定输入比较时，除热运行耗时与显存峰值外，还要逐帧观察细纹、移动的人脸与锐利边缘。[性能测量](../reference/benchmarks)给出了带条件的实测值。
