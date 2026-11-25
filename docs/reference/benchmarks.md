# Read the hardware measurements

<span id="current" aria-hidden="true"></span>
<span id="native544" aria-hidden="true"></span>
<span id="sm86-parallel" aria-hidden="true"></span>


## Source editing on 0.29.0: equal duration, different shot counts

Six recorded clips of about four seconds each were edited sequentially on one RTX PRO 6000 Blackwell Server. The workload used the original six-step Remake graph, silent output and a roughly 0.5 MP canvas (960 × 544, or portrait). Each time runs from job execution start to publication, including the job's model invocation, decode and output assembly. Procurement, cold startup, reference-image preparation and client download are outside this interval.

| Edit workload | Model windows and frames | Observed time |
| --- | --- | ---: |
| Three different continuous takes | 1 × 107 frames each | 62.9, 66.8, 67.5 s |
| Two shots | 2 × 90 frames | 117.4 s |
| Three shots | 3 × 90 frames | 202.9 s |
| Six short shots | 6 × 90 frames | 293.8 s |

This integration gave the observed shots separate windows. Even short shots require a minimum model window. To estimate work on your own footage, record the window count, model frames and delivered frames together. The six scenes had different composition and movement, so the times indicate an order of magnitude.

## Fixed-profile generation measurements

These complete-generation timings exclude model loading. The first three rows use the same 0.15 FP8 resident trunk with Veda 44/6, face repair and six-step audio repair on one RTX PRO 6000 Blackwell 96 GiB. They show the cost of changing one request dimension.

| GPU and recipe | Model canvas | Seconds of video | Elapsed |
| --- | --- | ---: | ---: |
| RTX PRO 6000, FP8 resident | 1920 × 1088 | 5 | 163 s |
| RTX PRO 6000, FP8 resident | 1920 × 1088 | 10 | 245 s |
| RTX PRO 6000, FP8 resident | 2720 × 1536 | 5 | 261 s |
| RTX 4090 48 GiB, 0.14 BF16 hybrid block ring | 2720 × 1536 | 5 | 659–672 s |

Peak device memory in the three PRO 6000 runs was 42.1 GiB. Four separate cold starts on that deployment took 75–76 seconds from provider start through model readiness. The 4090 row changes precision and placement as well as device. Use [hardware requirements](../guide/profiles) to select an allocation, then measure your own request from startup through the last encoded frame.

## Reproduce a comparison

Record the package and dependency versions, device count and interconnect, model and adapter revisions, prompt, seed, geometry, frame count and audio recipe. Separate cold load, warm inference, decode and encode. Compare the same source against the brief over multiple frames and listen to the soundtrack. [Timing methodology](./performance)
