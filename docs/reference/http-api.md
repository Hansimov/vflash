# Native HTTP response contract

This API accepts compiled conditioning bundles and returns native latent tensors. Python complete-pipeline
APIs produce MP4; `ReferenceAVRuntime` is not exposed here. The service owns one configured profile, serial
worker and bounded in-memory job history. Read `/openapi.json` from the actual server for the machine-readable
contract. Typed responses below are published in 0.22.1; older 0.18.0 container schemas remain unchanged.

| Request | Success | Other outcomes |
| --- | --- | --- |
| `GET /healthz` | 200: process status and input/output capabilities | Does not qualify assets, GPU or inference |
| `GET /readyz` | 200: ready, version, profile, checks, GPU, attention selection | 503: same structure with failed checks/error; model may still be cold |
| `GET /v1/profiles` | 200: the single active profile, NFE, precision, configured attention | A catalog entry does not change the loaded service |
| `POST /v1/denoise/jobs` | 202: a job record | 404 missing bundle; 422 invalid request/incomplete bundle; 429 queue full with Retry-After; 503 not ready/stopping |
| `GET /v1/denoise/jobs/{id}` | 200: current retained record | 404: unknown, expired or lost-on-restart record |
| `GET /v1/denoise/jobs/{id}/output` | 200 binary; 206 for a valid byte range | 409 not succeeded; 404 unavailable; 416 unsatisfiable range |

## Job lifecycle

Submit `{"bundle":"example-bundle"}`, relative to the configured read-only bundle root. Each accepted POST
creates a new id: **creation has no idempotency key**. A definitive 429 has not accepted work; wait for its
Retry-After. A timeout or lost response is uncertain and must not be blindly replayed. The calling application
owns durable submission/reconciliation; this API does not provide exactly-once recovery across restarts.

Job fields are id, status, profile_id, relative bundle, output_file, submitted_at, started_at, finished_at,
result and error. Timestamps are UTC ISO 8601 strings; not-yet-known values are null. Status transitions are
`queued → running → succeeded | failed`. Result contains profile-specific executor metadata and may gain
fields. It is not a video URL. Download from the output route after success.

An inference failure is a retained job with status failed and an error, returned by GET with HTTP 200.
Transport/admission errors instead use HTTP status and `{"detail": ...}`. Detail can be text, a readiness
object, or a list of field-validation problems. Do not assume every error has the same inner structure.
The server exposes `/docs` and `/openapi.json`; this local engine API has no Halo accounts, billing or API keys.
Keep it on its configured local binding or behind the owning application's authentication.

## Output and retention

`application/octet-stream` contains safetensors, not MP4. Stream the download to a file; Range supports
resuming byte transfer while the record and file remain available. Output_file is a server-owned filename,
not a caller-selected path. Completed history is bounded; pruning a record does not delete the file but removes
its API lookup/download. Restarts lose all in-memory records. Preserve completed outputs and receipts in the
calling application's storage. There is no cancellation or automatic replay endpoint.

## Version scopes

Use the [current Python/server installation](../guide/getting-started) for 0.22.1. The
[published container inventory](../guide/docker) still describes 0.18.0 images, whose original endpoints
continue working. Changing only documentation or package version does not qualify a different GPU/model.
