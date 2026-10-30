# 原生HTTP响应合同

本API接收编译好的条件bundle，返回原生潜变量。Python完整流水线交付MP4，`ReferenceAVRuntime`不通过
此HTTP接口提供。服务持有一个固定profile、串行worker与有界内存作业历史。机器可读合同以实际服务的
`/openapi.json`为准；下述响应类型在0.22.1发布，旧0.18.0镜像schema保持原样。

| 请求 | 成功 | 其他结果 |
| --- | --- | --- |
| `GET /healthz` | 200：进程状态与输入／输出能力 | 不代表资产、GPU或推理已通过 |
| `GET /readyz` | 200：ready、版本、profile、检查、GPU与注意力选择 | 503：相同结构中的失败检查／错误；模型仍可能尚未加载 |
| `GET /v1/profiles` | 200：唯一活动profile、NFE、精度与配置注意力 | 目录中的其他配置不会自动替换服务 |
| `POST /v1/denoise/jobs` | 202：作业记录 | 404缺bundle；422参数／bundle不完整；429队列满并含Retry-After；503未就绪／停止中 |
| `GET /v1/denoise/jobs/{id}` | 200：仍保留的作业记录 | 404：不存在、历史已淘汰或重启后丢失 |
| `GET /v1/denoise/jobs/{id}/output` | 200二进制；合法字节Range为206 | 409尚未成功；404不可用；416范围越界 |

## 作业生命周期

提交 `{"bundle":"example-bundle"}`，路径相对配置的只读bundle根目录。每个受理POST都会创建新id，
**没有创建幂等键**。明确429表示未受理，可按Retry-After等待；超时／响应丢失表示结果未明，不能盲目重放。
调用应用负责持久提交和核对，本API不提供跨重启的恰好一次恢复。

作业字段为id、status、profile_id、相对bundle、output_file、submitted_at、started_at、finished_at、
result和error。时间为UTC ISO 8601字符串，尚未知的值为null；状态为 `queued → running → succeeded | failed`。
result包含各profile的执行元数据，允许增加字段，并非视频URL。成功后通过output端点下载。

推理失败通过HTTP 200返回status=failed及error；传输／准入错误则用HTTP状态和 `{"detail": ...}`，
detail可能为文字、就绪对象或逐字段错误列表，客户端不能假设内部结构始终相同。
本地服务提供 `/docs` 和 `/openapi.json`，不包含Halo账号、计费或API密钥；保留本地监听，
或由所属应用提供访问认证。

## 输出与保留

application/octet-stream承载safetensors，不能直接当视频播放。流式保存到文件，作业记录和文件仍在时
可用Range续传。output_file由服务端决定，不接受客户端指定任意输出路径。
完成历史有界；淘汰记录不删除文件，但相应查询／下载URL失效。重启丢失全部内存作业记录，
调用应用应保存成品与回执。本API没有取消或自动重放端点。

## 版本范围

0.22.1通过[当前Python/server安装](../guide/getting-started)提供；
[已发布容器清单](../guide/docker)仍描述0.18.0镜像，原接口继续工作。
仅更新文档或包版本不能让另一GPU／模型自动获得资格。
