# Choose a runtime by input and output

| What you have | What you need | Interface |
| --- | --- | --- |
| Text, a first or last frame, or ordered references | A local MP4 with a fixed execution profile | [H3Pipeline or the CLI](./complete-pipeline) |
| Text or a first frame and prepared Pika weights | Six visual evaluations with joint sound or a separate Larry audio pass | [PikaAVRuntime](./pika-flex) |
| Photos of the subjects | A new shot preserving their appearance | [PikaReferenceAVRuntime](./pika-reference-weights) |
| Existing footage and a target appearance | An edited window on the source clock | [Remake](./remake) |
| Compiled conditioning and a matching native profile | Latents for your own decoder | [Native HTTP API](../reference/http-api) |

## Choose what a reference controls

A first frame fixes the opening composition. Two endpoint frames fix the opening and the ending. An appearance reference supplies a subject, clothing or product design and leaves the composition open. For a change such as a new watch dial **on the same hand movement**, start with the source clip and edit its visible dial.

## Plan device and asset ownership

Choose a [tested GPU configuration](./profiles) for that specific runtime. The prepared Pika and original BF16 Pika runtimes have different assets and Python environments. For a service that repeatedly uses a single model, allocate the device, prepare assets once and reuse the [session](./python). The calling application stores the user's originals, completed films and job history.
