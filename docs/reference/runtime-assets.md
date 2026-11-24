# Understand the four native inputs

| Input | Files | Lifetime |
| --- | --- | --- |
| Transformer artifact | `artifact.json` and compiled blocks | Per model and GPU target |
| Schedule overlay | `overlay.json` and schedule tensors | Per sampling recipe |
| Auxiliary projections | Auxiliary safetensors file | Per compatible model revision |
| Conditioning bundle | `bundle.json`, conditioning tensors and declared inputs | Per request |

## Reuse model resources

Mount the first three read-only and load them once per session. Give each distinct request a separate bundle. Preserve the order, type, geometry and identity of reference assets when preparing it. See [compile weights](../guide/compile-weights) for the complete-pipeline path and [HTTP](./http-api) for submission by bundle name.


<span id="versions" aria-hidden="true"></span>
<span id="video-conditioning" aria-hidden="true"></span>

## Diagnose a mismatch

Compare the profile’s target, adapter revision and schedule with the compilation receipt. A four-step and an eight-step adapter require matching schedules. Keep a verified inventory at ingestion; replacing a file under an old receipt invalidates that evidence. Keep generated outputs in a separate writable directory.
