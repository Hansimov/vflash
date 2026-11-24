# Continue from native audio and video latents

Use these Python interfaces to retain the model output before media encoding and to supply compatible history to the next request. The overlap, source cut and audio clock must agree across both windows.

## Retaining native audio/video latents

Python callers may pass `latent_output=Path("clip.safetensors")` to `H3Pipeline.generate`.
The destination must be distinct from the MP4 and must not exist. The complete native AV output
is published atomically before media decoding; it remains available if decoding or encoding later
fails. Successful `VideoResult` objects report the retained path. Omit the option for the existing
temporary-file lifetime. Retention does not itself retry decoding or qualify a continuation method.

## Experimental native AV target prefixes

`H3Pipeline.generate(..., native_av_prefix=prefix, latent_output=path)` accepts an advanced
`vflash.native.h3_av_continuation.NativeAVPrefix`. Construct it with `from_latents`, supplying
native `video_latents` / `audio_latents`, the selected `source_frames`, `context_frames`, and
51 clean-timestep modulation tensors compiled from the same model weights (50 block tensors
at timestep 1.0 and the final-layer tensor). These are explicit caller-owned assets; the engine
does not download or infer a different checkpoint. Python only; CLI/HTTP admission is unchanged.

The current base execution scope is four-step I2VA or FL2VA on the single SM89 Veda block ring,
with one or two same-canvas keyframes. Repaint and profiling are rejected; optional native audio
refinement must explicitly preserve matching history as described below. Both source cuts and overlap lengths must be `17n+5` frames. The overlap must
also be divisible by 3 (for example 39 or 90), so it has an exact 40Hz audio duration. Source
cuts may use any aligned frame, including 107 and 124; their nearest audio tick differs by
at most 1/120 second. The first image represents the
start of the context, not its last frame. The context occupies the generated target's beginning;
remove that overlap in latent space and jointly decode the assembled sequence. A returned
five-second MP4 still contains the context and must not be appended wholesale.

`prefix.join(source_latents, generated_latents, generated_frames=124)` verifies the exact
preserved prefix, removes the overlap and returns the joined latent dictionary. It rounds the
audio length from the complete sequence clock, trimming at most one excess tail tick and
rejecting missing audio. This prevents rounding drift across many continuations. Retain the
joined frame count (`source_frames + generated_frames - context_frames`) with the artifact.

The scope preserves video at conditioning strength 0.999 and audio at 1.0 during each Euler
step, then restores exact clean AV context. Modulation and row plans belong to one request
and are restored on exit. A complete 960×544, five-second SM89 execution matched the tested
scoped implementation's native video and audio exactly. This establishes integration parity;
physical motion and subjective soundtrack quality remain output-specific evaluations.

For an intentional scene cut, construct the prefix with `preserve_video=False`. The native
audio context is still restored at each step, while every target picture row is generated
under the new scene's keyframe. The current same-canvas and execution restrictions still apply.
Discard the target's first `context_frames` pictures as preroll, and call
`prefix.join_audio(source_latents, generated_latents, generated_frames=124)` to obtain the
joined audio tensor for a single decode. Cut the pictures independently on that frame clock;
`join` deliberately rejects this mode because it would claim video continuity. The record
reports zero preserved video rows and `video_prefix_exact=None`. Two complete SM89 scene-cut
trials preserved all 65 history ticks with a 39-frame preroll. They establish execution and
clock correctness; music continuity, speech and synchronization require listening to the result.

If the delivered sequence intentionally ends before its model-padding tail, pass that exact
frame count as `join_audio(..., delivery_frames=N)`. Two independently rounded durations can
leave the unused joined tail one audio tick short. This option verifies that all delivered
sound exists and retains available native padding for decoding; trim the decoded waveform
to the declared delivery clock. It never pads missing audio. Omitting the argument still
requires the complete joined model clock, including its padding.

Prepare those tables with `vflash.compiler.continuation.compile_native_history_tables`.
Pass the local base transformer checkpoint; for a hybrid, also pass its local reference
checkpoint and exact `reference_blocks` tuple, such as `tuple(range(25, 50))`. The helper
reads only those tensors, computes all 51 clean-context tables on CPU and returns a tuple
for `NativeAVPrefix.from_latents`. Reuse it for that model selection. It leaves weights
and global thread settings unchanged; an optional `progress(completed, total)` callback
can enforce the caller's deadline. The native-prefix interfaces are experimental Python APIs included in 0.22.0;
0.21.0 and earlier wheels do not contain them.


To refine sound while preserving that continuation, compile the refinement and its
history tables together with `compile_refinement_tables(..., include_native_history=True)`.
Passing `reference_directory=None` explicitly selects original FL for all50 blocks; an
explicit reference checkpoint retains the existing FL0–24/Ref25–49 recipe. The `source`
field records that choice. Audio mode alone does not select pure FL.
Load the companion tables with `load_refinement_history_tables(directory)` from
`vflash.native.h3_refinement_schedule`, then construct an audio-only `NativeAVPrefix`
from the same source window and set `AudioRefinementRequest.native_audio_prefix`.
Do not mix pure-FL clean history tables with hybrid refinement tables. The loader checks
the companion asset's declared source, inventory, geometry and finite BF16 values.
The complete pipeline checks the main/history source window and audio equality before
loading. For retained outputs, pass `EncodedAudioSource` and this refinement to the native
session, avoiding another main generation or video decode.

Every refinement evaluation reads and restores the clean audio prefix at timestep1.0.
The entire video remains frozen; only later audio is refined. Modulation storage expands
before the optional weight cache is built, and the cache is released before restoring
original tables. This path uses serial ring execution and disables video-decode overlap.
Failures restore the original audio, video, schedule and owned modulation resources.
Initial retained nine/ten-second SM89 executions preserved all65 context ticks, every
video latent and the encoded picture streams in joined ten/eighteen-second deliveries.
A subsequent asset audit found hybrid refinement tables paired with pure-FL history tables
in that trial. It is retained as a failed recipe, not a correctly matched audio qualification.
Matching bundle compilation prevents that source-selection mistake. Four subsequent SM89
executions compared matched hybrid and original-FL twelve-step recipes on two retained targets.
All kept the65-tick history and video bits. Each cached27 BF16 blocks under a20 GiB budget
with0.25MP spatial audio context. This qualifies the combined execution path, without a
GPU cache-equivalence or perceptual audio-quality claim. Six/twenty-step history variants
currently have CPU ownership tests only.


FL2VA additionally accepts the desired final image while preserving the same target history.
Both conditioning keyframes precede the target rows; preservation offsets by their complete
row count and does not overwrite either condition. One complete fifteen-second SM89 FL2VA
target preserved all39 video frames and65 audio ticks, joining with its source into396 frames.
The endpoint guided the eventual pose and framing in that trial; repeated movement and a
long mid-shot hold remained. End-state constraints do not certify natural motion or pacing.
History-aware audio refinement currently retains its I2VA-only admission.


The same `delivery_frames` override is available on `join`. If the last delivered video
frame extends beyond the nearest40Hz tick, the join retains the available real tick
covering that frame. For example, joining124+124−39 frames needs349 audio ticks for
all209 video frames, although the nearest-clock count is348. This preserves real decoder
padding instead of validating349 ticks and then cropping one away. Missing required
audio is still rejected; the default nearest-clock join remains unchanged.
