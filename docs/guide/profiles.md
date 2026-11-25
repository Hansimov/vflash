# Hardware requirements

The table is ordered by the reference model’s peak **single-GPU FP32** throughput. The memory column describes the boards tested with Vflash. Tensor precision, memory bandwidth, model placement and inter-GPU links also affect film delivery time.

<span id="heterogeneous" aria-hidden="true"></span>

| GPU | Reference FP32 | Tested memory | Workload and limits |
| --- | ---: | --- | --- |
| RTX PRO 6000 Blackwell Server | 120 TFLOPS | 96 GiB | Prepared Pika, original Remake and resident hybrid. The prepared AV recipe requires a full device. |
| RTX 4090 | 82.6 TFLOPS | 48 GiB | SM89 fixed profiles for text, references and keyframes. Check free memory before selecting a resident or HD path. |
| H100 SXM | 67 TFLOPS | 80 GiB | SM90 hybrid first-frame generation. A resident trunk requires at least 75 GiB. |
| Two RTX 3080 | 29.8 TFLOPS per reference card | 2 × 20 GiB | SM86 block streaming. Both cards cooperate on one request; allow at least 64 GiB of available host RAM per worker for the measured native workload. |

The 4090 and 3080 specification figures come from NVIDIA’s reference boards. Published Vflash runs used higher-memory configurations. Check your actual allocation with `vflash doctor`. Each card in a pair retains its own memory limit.

For source editing, compare the [0.29.0 measurements by shot count](../reference/benchmarks) on a single card and canvas class.

## Select a profile {#available}

Run `vflash profiles` to list identifiers and `vflash plan PROFILE --gpu 0` to resolve placement. On a matching pair, add `--peer-gpu 1 --strategy sequence-head`. For a resident hybrid on H100 or Blackwell, verify the [profile and allocator requirements](./pipeline-conditioning) before loading weights.

## Sources

Peak specifications come from [RTX PRO 6000 Server](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [Ada architecture, appendix A](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-V2.02.pdf), [H100 SXM specifications](https://www.nvidia.com/en-us/data-center/h100/) and [Ampere GA102, RTX 3080](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf). For elapsed time on a particular workload, see [measured runs](../reference/benchmarks).

## Resident and smaller-memory configurations

<span id="h100-resident" aria-hidden="true"></span>
<span id="sm120-resident" aria-hidden="true"></span>
<span id="memory" aria-hidden="true"></span>
<span id="small-blackwell" aria-hidden="true"></span>
<span id="lora" aria-hidden="true"></span>

A resident trunk needs headroom for encoding and decoding. Consult the [versioned hardware measurements](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/guide/profiles.md) for bounded H100 NVL and RTX 5090 trials, adapter choices and placement settings.
