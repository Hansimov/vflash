# 参与开发

Vflash 是可复用的推理引擎。产品账户、计费、私有提示词、机群凭据和网站代码都不属于本仓库。修改运行时
边界前，请阅读 [`AGENTS.md`](https://github.com/Hansimov/vflash/blob/main/AGENTS.md) 的最新职责地图。

## 开发环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,server]'
pre-commit install

pytest
pre-commit run --all-files
npm ci
npm run docs:build
```

默认测试在收集用例前隐藏CUDA设备。硬件测试必须设置`VFLASH_TEST_CUDA=1`，并通过
`CUDA_VISIBLE_DEVICES`明确选择已分配的单卡或卡组，在独立进程中执行；模型文件和生成媒体不能进入Git。

## 一次只改变一个合同

从固定 profile、artifact 和请求开始，明确改动属于数值算术、内存布局、加载、调度、媒体交付还是诊断。
不要在 kernel 性能对照里同时改变 adapter、步数或精度。

精确布局优化应先在代表性shape证明逐元素相等，再运行完整请求。近似必须具名并报告非精确，不能通过
依赖环境的静默fallback引入。0.5.0明确将单SM89 Base16默认改为Sol，保留prepared模型身份及显式dense选项；
这是公开的版本行为变更，不代表完整媒体质量门关闭，也不允许顺带引入其他未验证的近似。

适用时增加 fail-closed 输入检查和 CPU 参考路径。运行时 fallback 必须在结果元数据中可见，或者严格限制为
保持语义的实现。

## 目标硬件资格

SM86 与 SM89 是两个独立目标，在一种架构通过不能替另一种完成资格验证。需要记录：

- 精确源码 revision、依赖版本和 profile 身份；
- GPU 型号/数量和拓扑类别，但公开材料中不放设备 UUID；
- 冷准备和重复请求耗时分开；
- 同一卡组上的 A/B/A 或重复测量；
- 设备/主机内存峰值、持续温度和实际热降频；
- 完整输出、清理与取消行为；
- 所有算术近似的解码多帧和音频证据。

kernel 计时只能证明机制，不能直接成为端到端结论。双卡单请求延迟与相同卡数的 replica 吞吐必须分开。

## 公开证据与隐私

提交代码、测试、可复现的公开命令和聚合证据。不要提交凭据、私有媒体、提示词、服务配置、主机名、设备
UUID 或应用专用存储路径。迭代时避免反复哈希大型模型文件，应使用固定 source manifest 和一次性准备 receipt。

实现受到上游工作的启发时，应固定复核版本、保留许可证并明确致谢；文档要区分已采用和仍未合格的机制。

## 发布检查

打 tag 前：

1. 运行聚焦测试、完整 CPU suite、隐私 hook 和双语文档构建；
2. 从 clean checkout 安装包并重跑公开 CLI smoke；
3. 确认 README、Pages、CLI profile 和版本元数据描述同一边界；
4. 记录目标硬件证据，不声称未测试架构或 shape；
5. 对一个不可变 commit 打 tag，并在 release notes 区分精确、近似与实验行为。

模型 checkpoint 继续遵循各自许可证，且永远不作为 release asset 发布。
