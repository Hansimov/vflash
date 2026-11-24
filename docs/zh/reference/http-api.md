# 原生 HTTP 接口

服务接收相对条件包名称，返回含视频与音频潜变量的 safetensors。配置在启动时确定；当前服务的精确字段以其 `/openapi.json` 为准。

## 提交与下载

| 接口 | 成功响应 |
| --- | --- |
| `GET /healthz` | 200，进程状态 |
| `GET /readyz` | 200，配置、设备和就绪检查 |
| `GET /v1/profiles` | 200，当前启用的配置 |
| `POST /v1/denoise/jobs` | 202，作业 ID 和记录 |
| `GET /v1/denoise/jobs/{id}` | 200，当前状态及执行错误 |
| `GET /v1/denoise/jobs/{id}/output` | 200 返回文件，206 返回请求的字节范围 |

保存 ID，沿用同一 ID 查询 `queued` 和 `running`。进入 `succeeded` 后下载；HTTP 查询成功时，记录本身仍可能显示 `failed`。

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/denoise/jobs \
  -H 'content-type: application/json' \
  -d '{"bundle":"example-bundle"}'
```

```bash
curl -fsS http://127.0.0.1:8000/v1/denoise/jobs/JOB_ID
```

```bash
curl -fLo result.safetensors \
  http://127.0.0.1:8000/v1/denoise/jobs/JOB_ID/output
```

## 按结果恢复

| 状态 | 含义与处理 |
| --- | --- |
| 404 | 条件包、历史作业或输出不可用。检查挂载及原作业 ID。 |
| 409 | 输出尚未完成。先读取作业状态。 |
| 416 | 请求范围超过文件边界。修正起始字节。 |
| 422 | 请求或条件包未通过校验。修改报错字段或资源。 |
| 429 | 队列已满，本次未受理。按 `Retry-After` 等待。 |
| 503 | worker 不可用。提交前检查 `/readyz`。 |

创建接口没有幂等键。提交响应丢失时先核对受理情况，再决定是否重发。服务重启后，内存中的作业索引会清空。
