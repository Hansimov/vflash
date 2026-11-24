# Native HTTP API

The service accepts a relative bundle name and returns safetensors with video and audio latents. Configure its profile at startup. Query the running server’s `/openapi.json` for its exact schema.

## Submit and fetch

| Endpoint | Successful response |
| --- | --- |
| `GET /healthz` | 200, process status |
| `GET /readyz` | 200, profile, device and readiness checks |
| `GET /v1/profiles` | 200, active profile |
| `POST /v1/denoise/jobs` | 202, job ID and record |
| `GET /v1/denoise/jobs/{id}` | 200, current state and any execution error |
| `GET /v1/denoise/jobs/{id}/output` | 200 for the file or 206 for a requested byte range |

Poll the same ID through `queued` and `running`. Download only after `succeeded`. A retained record can say `failed` in a successful HTTP response.

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

## Recover by outcome

| Status | Meaning and next action |
| --- | --- |
| 404 | The named bundle, retained job or output is unavailable. Check the mount and original ID. |
| 409 | Output is unfinished. Read the job state before downloading. |
| 416 | The range exceeds the file. Correct the byte offset. |
| 422 | Request or bundle validation failed. Correct the indicated field or input. |
| 429 | The queue refused admission. Wait for `Retry-After`. |
| 503 | The worker is unavailable. Inspect `/readyz` before submitting. |

Creation has no idempotency key. If a submission response is lost, investigate acceptance before replaying it. Restarting the server clears the in-memory job index.
