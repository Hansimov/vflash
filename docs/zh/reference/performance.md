---
search: false
---

# 测量一次完整推理

<span id="parallel" aria-hidden="true"></span>
<span id="quality" aria-hidden="true"></span>


分别测量进程启动后的首个请求和同一进程连续生成的热请求。记录主机与设备内存峰值、输入编码、去噪、音频精修、解码及 MP4 写出耗时。单个内核的加速幅度要结合其占总耗时的比例来解释。同时在指定时间点比较输出画面与声音。[硬件实测](./benchmarks)给出了完整例子。

[查看此实现的详细配方与测量记录](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/zh/reference/performance.md)
