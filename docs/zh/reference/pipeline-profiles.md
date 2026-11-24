# 选择匹配的模型配置

<span id="lightx-v01" aria-hidden="true"></span>
<span id="sm86" aria-hidden="true"></span>


固定配置绑定主干模型族、适配器、步数、精度与 GPU 架构。运行 `vflash profiles` 得到已安装版本的完整列表，随后用其中的标识规划设备及准备资源。

## 按输入选模型族

| 输入 | 配置族 | 示例 |
| --- | --- | --- |
| 文字 | T2VA Turbo4 | `t2va-turbo4-exact-sm89` |
| 外观参考图 | Ref2VA Turbo4 | `ref2va-turbo4-exact-sm89` |
| 首帧、尾帧或首尾帧 | Base16 或 LightX 关键帧 | `i2va-base16-bf16-sm89` |
| Pika 联合音画 | 准备权重的运行时配方 | [Pika 音画](../guide/pika-flex) |
| 原片与目标外观 | Remake edit 或 swap 权重 | [编辑源视频](../guide/remake) |

目录标识用于固定配置；Pika 配方与 Remake 权重分别通过自身接口选择。

## 加载前检查

先在实际硬件上解析配置。主干产物、采样时序及辅助张量应属于同一配置。需要切换模型族时重新准备匹配权重。完整示例见[权重编译](../guide/compile-weights)。

```bash
vflash profiles
vflash plan ref2va-turbo8-exact-sm89 --gpu 0
```
