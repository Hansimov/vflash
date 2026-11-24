# Choose compatible model profiles

<span id="lightx-v01" aria-hidden="true"></span>
<span id="sm86" aria-hidden="true"></span>


A fixed profile binds a transformer family, adapter, step schedule, precision and GPU architecture. Generate the list for your installed version with `vflash profiles`. Use the identifier it prints for planning and preparation.

## Pick the model family

| Input | Profile family | Example |
| --- | --- | --- |
| Text | T2VA Turbo4 | `t2va-turbo4-exact-sm89` |
| Appearance references | Ref2VA Turbo4 | `ref2va-turbo4-exact-sm89` |
| First, last or endpoint frames | Base16 or LightX keyframes | `i2va-base16-bf16-sm89` |
| Joint Pika audio/video | Prepared runtime recipe | [Pika AV](../guide/pika-flex) |
| Original film and a target appearance | Remake edit or swap weights | [Edit a source video](../guide/remake) |

A catalog identifier selects fixed-profile execution. Pika recipes and Remake weights are selected through their own constructors.

## Validate before loading

Resolve the profile on the actual hardware. Match the compiled artifact, schedule and auxiliary tensors to the same profile. Renaming a directory cannot convert one family into another. See [weight compilation](../guide/compile-weights) for a complete example.

```bash
vflash profiles
vflash plan ref2va-turbo8-exact-sm89 --gpu 0
```
