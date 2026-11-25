# 使用 Docker 运行 Vflash

先选择容器要交付的内容。原生服务读取挂载的条件包，返回潜变量；pipeline 镜像使用已准备的模型资源生成 MP4。

| 镜像 | 已发布的 linux/amd64 标签 |
| --- | --- |
| Native HTTP | `hansimov/vflash:0.21.0` |
| Pipeline | `hansimov/vflash:0.21.0-pipeline` |

部署时请使用[清单中的镜像摘要](images.json)固定构建。需要 0.29.0 包的功能时，按[部署手册](https://hansimov.github.io/vflash/zh/guide/docker)从对应源码标签构建。模型资源只读挂载，单独分配可写输出目录。发送任务前检查就绪状态，并在容器内核对包版本。
