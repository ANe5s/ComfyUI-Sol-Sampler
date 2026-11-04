# ⚡ ComfyUI-Sol-Sampler

[简体中文](README.zh.md)

## Sol-Attn Progressive Sampler — Lightning-Fast Progressive Sampling

**Sol-Attn Progressive Sampler** is a sampling strategy for high-resolution video generation with ComfyUI's native **Model Sparse Attention (Sol-Attn)**. It runs low-resolution denoising, spatial latent upscaling, and high-resolution denoising as one progressive sampler execution.

## The discovery

This idea began with a problem observed while using the SelfLift sampler: chunked denoising could isolate context between chunks. Experiments with global context showed that removing denoising chunks made generation faster, while also resolving the memory problem that chunking was meant to address. Inspection of node behavior, code, and further experiments pointed to Sol-Attn; Spark-H3 was then tested with this approach.

## How it works

Sol-Attn moves from dense attention during warm-up toward sparse attention. Dense attention has higher compute and memory costs; as attention becomes sparser, those costs fall. Under high memory pressure, the initial warm-up can be slow when dense attention, latent activations, and model weights all compete for resources at once.

The sampler separates that peak in time and coordinates two changing memory costs:

```text
Sol-Attn:     dense warm-up  →  progressively sparse  →  sparse attention
Video latent: low-resolution latent → latent upscale → high-resolution latent
```

The low-resolution phase lets Sol-Attn warm up while the latent is small. After the attention context is warmed and sparsity has increased, the sampler upscales the latent and continues denoising at high resolution in the same sampling run. The larger latent therefore enters when attention requires less memory, instead of overlapping its peak with dense-attention warm-up and model loading.

Extensive ComfyUI generation runs have validated the speed and memory benefits of this approach in practice. The optimization is built around two elements: Sol-Attn and a joint low-/high-resolution sampling plan that Sol-Attn treats as one sampling run. It avoids chunked denoising, removing the context isolation that motivated the original investigation. The optional H3 temporal chunking setting applies only inside latent upscaling; it does not split the denoising stages.

The sampler splits the input sigma schedule at `lowres_steps`. It denoises the latent at reduced spatial resolution, upscales the denoised latent, reconstructs the high-resolution state at the transition sigma, then continues with the remaining sigmas. The same selected native ComfyUI sampler is used in both stages; the node does not add a second noise schedule or extra denoising steps. Actual peak usage varies with the model, latent shape, attention implementation, and other models resident in memory.

## My test environment and generation records

- GPU: NVIDIA GeForce RTX 5070 (12 GB VRAM)
- CPU: AMD Ryzen 5 9600X
- System memory: 32 GB

All runs use 16:9 and `lowres_scale=0.5`; the `ResolutionSelector multiple` setting varies by workflow. The table lists the acceleration LoRA used for each record; some runs also include additional style LoRAs. The FL2VA checkpoint is `minimax_h3_fl2va_pruned_int8_convrot.safetensors`; the REF2VA checkpoint is `minimax_h3_ref2va_pruned_int8_convrot.safetensors`.

`Sampler` is `[Sol-Sampler] Sampling completed`; `Full workflow` is ComfyUI's `Prompt executed`. Times are seconds. Output filenames, dimensions, frame counts, and durations come from local MP4 metadata. The videos remain in the local ComfyUI output folder and are not included in this repository.

### FL2VA model

- **DMAD/larryvrhv4 merge:** `minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors` at 0.8–1.0; Euler / Beta; 10 steps, `lowres_steps=6`.
- **Pika:** `pika_h3_flex_fl2va_comfy_bf16.safetensors` at 1.0; LCM / Simple; 9 steps, `lowres_steps=5`.

| Acceleration LoRA | Target | Sampler | Full workflow | Notes |
| --- | --- | ---: | ---: | --- |
| DMAD/larryvrhv4 merge | 1.0 MP / 15 s | 375.192 s | 425.82 s | 1376 × 768, 362 frames, 15.083 s |
| DMAD/larryvrhv4 merge | 1.5 MP / 10 s | 383.567 s | 443.40 s | 1664 × 928, 243 frames, 10.125 s |
| DMAD/larryvrhv4 merge | 2.0 MP / 8 s | 506.743 s | 568.42 s | 1920 × 1088, 192 frames, 8.000 s; `chunks=8`, `seq_threshold=2048` |
| DMAD/larryvrhv4 merge | 3.0 MP / 5 s | 478.564 s | 535.06 s | 2368 × 1344, 124 frames, 5.167 s; `chunks=8`, `seq_threshold=2048` |
| Pika | 1.0 MP / 15 s | 408.283 s | 477.36 s | 1360 × 768, 362 frames, 15.083 s |
| Pika | 1.5 MP / 10 s | 418.273 s | 484.01 s | 1680 × 944, 243 frames, 10.125 s |
| Pika | 2.0 MP / 8 s | 495.654 s | 563.76 s | 1936 × 1088, 192 frames, 8.000 s |
| Pika | 3.0 MP / 5 s | 498.364 s | 549.58 s | 2368 × 1344, 124 frames, 5.167 s; `chunks=8`, `seq_threshold=2048` |

### REF2VA model

- **Checkpoint:** `minimax_h3_ref2va_pruned_int8_convrot.safetensors`.
- **DMAD/larryvrhv4 merge:** `minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors` at 0.8–1.0; Euler / Beta; 10 steps, `lowres_steps=6`.
- **Pika:** `pika_h3_flex_ref2va_comfy_bf16.safetensors` at 1.0; LCM / Simple; 9 steps, `lowres_steps=5`.

| Acceleration LoRA | Target | Sampler | Full workflow | Notes |
| --- | --- | ---: | ---: | --- |
| DMAD/larryvrhv4 merge | 1.0 MP / 15 s | 510.795 s | 601.00 s | 1376 × 768, 362 frames, 15.083 s |
| DMAD/larryvrhv4 merge | 1.5 MP / 10 s | 564.405 s | 667.00 s | 1664 × 928, 243 frames, 10.125 s |
| DMAD/larryvrhv4 merge | 2.0 MP / 8 s | 1256.599 s | 1440.00 s | 1936 × 1088, 192 frames, 8.000 s; `chunks=8`, `seq_threshold=2048` |
| DMAD/larryvrhv4 merge | 3.0 MP / 5 s | 434.076 s | 500.77 s | 2368 × 1344, 124 frames, 5.167 s |
| Pika | 1.0 MP / 15 s | 560.169 s | 730.00 s | 1360 × 768, 362 frames, 15.083 s |
| Pika | 1.5 MP / 10 s | 426.371 s | 497.48 s | 1664 × 928, 243 frames, 10.125 s |
| Pika | 2.0 MP / 8 s | 486.710 s | 557.45 s | 1920 × 1088, 192 frames, 8.000 s; `chunks=8`, `seq_threshold=2048` |
| Pika | 3.0 MP / 5 s | 736.431 s | 803.00 s | 2368 × 1344, 124 frames, 5.167 s; `chunks=8`, `seq_threshold=2048` |

My broader local runs have appeared to show T2VA and REF2VA taking relatively similar time. This strict comparison is still incomplete: only one matched REF2VA/Pika row finished, and these measurements were made with other desktop activity running. I will add the remaining timings after direct runs under lower background load.
## Requirements

- ComfyUI with its native **Model Sparse Attention (Sol-Attn)** nodes.
- A compatible image or video latent workflow. Spatial video latents are upscaled; non-spatial streams such as audio are kept at their original shape.
- For learned upscaling, a MiniMax H3 3D latent upscaler checkpoint (`.safetensors` or `.pth`).

## Installation

Place this folder in `ComfyUI/custom_nodes/ComfyUI-Sol-Sampler`, then restart ComfyUI.

For **Minimax H3 3D** upscaling, place the checkpoint in:

```text
ComfyUI/models/latent_upscale_models/
```

Restart ComfyUI after adding a checkpoint so it appears in the node's model list.

## Workflow

1. Apply ComfyUI's native **Model Sparse Attention (Sol-Attn)** to the diffusion model.
2. Connect that model to the `MODEL` input of your guider, such as `CFGGuider`.
3. Connect the guider to `SamplerCustomAdvanced`.
4. Connect **Sol-Attn Progressive Sampler** to `SamplerCustomAdvanced`'s `sampler` input.
5. Connect the usual noise, sigma schedule, and latent inputs to `SamplerCustomAdvanced`.

The node is in **ANe5s Nodes → Sol-Sampler** and outputs a `SAMPLER`.

## Joint optimization strategy

The [showcase workflows](examples/) combine attention, sampling, and memory-management optimizations.

### Attention and memory path

Apply model processing in this order:

`Model Attention Backend` → `Model Sparse Attention (Sol-Attn)` → `MiniMax H3 Chunk FeedForward`

`MiniMax H3 Chunk FeedForward` defaults to `chunks=2` and `seq_threshold=4096`. If it becomes a performance bottleneck, try increasing `chunks` to `4`, `8`, or `16`, and lowering `seq_threshold` to `2048`.

For medium-to-high memory pressure, I recommend KJ/VRAMdebug cleanup nodes to release VRAM and host-memory usage. Place them at latent handoffs and before the final video save.

### Sampling path

Apply ComfyUI's native `Model Sparse Attention (Sol-Attn)` to the model, then connect `Sol-Attn Progressive Sampler` to the `sampler` input of `SamplerCustomAdvanced`. Sol-Attn progressively sparsifies attention and provides attention-level VRAM relief; the progressive sampler coordinates low-resolution denoising, latent upscaling, and high-resolution denoising as one sampling run. This serves the memory goal often sought with chunked denoising while preserving a shared sampling context.

Keep `upscaler_temporal_chunking` enabled. It is on by default and applies to the latent upscaler. The implementation prioritizes broad device compatibility, so occasional upscaling issues may still depend on the hardware and runtime environment.

## Why the showcase workflows use Pika H3 Flex

This discussion is framed by the related research I made public on and after August 7, 2026. With the larryvrh V1 step500 Turbo LoRA, I observed a characteristic that may be related to the model or training method: during the early training stage of the larryvrh Turbo LoRA, its acceleration path had not yet shown obvious overfitting. When combining the base model with the LoRA and denoising along the full sigma trajectory, the denoising directions and characteristics in the early high normalized-sigma range and the later low normalized-sigma range are clearly different. The high-sigma stage mainly affects global structure, while the low-sigma stage mainly affects detail. This shows that one Turbo LoRA can retain distinct high- and low-noise characteristics across normalized sigma ranges.

These characteristics disappeared in later accelerated models. Later larryvrh and LightX2V versions did learn to approximate the target denoising trajectory in fewer steps, but this high-/low-noise behavior disappeared in subsequent accelerated models. For accelerated LoRAs labeled for 4 or 8 steps, even increasing the sampling steps and extending the low-sigma phase did not make the additional denoising steps improve the image. The oily, AI-smeared appearance remained similar to the 8-step result, as if the acceleration path had locked the image-quality ceiling.

After Civitai creator tsolful merged DMAD with larryvrh's v4_step600, this issue was resolved and overfitting in the LoRA acceleration path was alleviated. Following extensive testing and workflow sharing by Bilibili creator YZ_金鱼, `minimax_h3_dmad_full_dareties_v4_step600_comfy` gradually became widely used for second sampling because of its better results. This weight merge also has a tradeoff: some objects, such as mechanical forms, may become underfit in shape and suffer structural breakdown. See tsolful's model page.

After Pika H3 Flex was released, its six-step acceleration LoRA showed the acceleration-path underfitting I was familiar with. I brought this LoRA back into my tests and observed the familiar high-/low-noise behavior. My experiments suggest the following: the LCM sampler's update rule is relatively fixed, but the actual denoising path is still affected by the sigma sequence. In common second-sampling workflows, directly reusing the officially recommended fixed six-step sigma schedule causes a number of problems. My tests indicate that nine or more steps are more suitable. When generating sigmas with a scheduler, prefer Simple, followed by Beta. Five low-resolution steps are necessary: if their sigma sequence does not cover the normalized high-sigma range, some objects may disappear completely; if too few steps are assigned to the low-sigma range, fast-moving objects may be underfit.

### Pika sampling settings

| Setting | Selection |
| --- | --- |
| Sampler | `lcm` for both FL2VA and REF2VA. |
| Scheduler | `Simple` first, followed by `Beta`. |
| Total steps | Nine or more; the showcase workflows use nine. |
| `lowres_steps` | `5`; with nine total steps, the high-resolution stage has four steps. |
## Inputs

| Input | Default | Description |
| --- | --- | --- |
| `sampler_name` | `euler` | Native ComfyUI sampler used by both resolution stages. |
| `lowres_steps` | `6` | Number of denoiser evaluations at low resolution. At least one step is reserved for the high-resolution stage. |
| `lowres_scale` | `0.5` | Spatial scale for the low-resolution latent, relative to the input latent. Set to `1.0` to run a single sampling pass without latent upscaling or a second stage. |
| `lowres_noise_mode` | `2D orthonormal Haar LL noise downsampling` | Maps the input high-resolution noise to the low-resolution stage. It sums each non-overlapping 2×2 spatial block and divides by 2; for iid standard Gaussian noise, this preserves unit variance. It requires the low-resolution latent to be exactly half-size in both spatial dimensions. Select `Bilinear noise downsampling` for other scale ratios. |
| `latent_upsample` | `Minimax H3 3D` | Use the learned H3 upscaler, bilinear interpolation, or nearest-neighbor interpolation. |
| `upscaler_model` | First available H3 checkpoint, or `none` | H3 checkpoint in `models/latent_upscale_models`. If `Minimax H3 3D` is selected with `none`, bilinear interpolation is used. |
| `upscaler_precision` | `checkpoint` | Inference dtype for the H3 upscaler: checkpoint dtype, FP32, FP16, or BF16. |
| `upscaler_temporal_chunking` | `true` | Processes long video latents in overlapping temporal chunks to reduce peak working memory. Audio latents are not chunked. |
| `upscaler_layerwise_cpu_offload` | `false` | Keeps H3 weights on CPU and moves modules to the GPU for their forward pass. This reduces the upscaler's peak VRAM use at the cost of CPU/GPU transfer time. |

When `lowres_scale` is `1.0`, the upscaler settings are unused. If the workflow has fewer than two sampling steps, the node also falls back to a single pass because there would be no high-resolution step to run.

## Upscaler memory behavior

The H3 upscaler is cached in CPU memory for reuse. In standard model-management mode, after the latent transition the node asks ComfyUI's model manager to unload the upscaler weights from managed GPU memory. With layerwise CPU offload enabled, each module is returned to CPU after its forward pass instead. In both modes the upscaler weights remain cached in host RAM; the upscaled latent remains available for the high-resolution sampling stage.

## License

Original project code and documentation are licensed under the Apache License 2.0; see [LICENSE](LICENSE). The `h3_latent_upscaler.py` file, which contains code adapted from upstream, is distributed under the upstream MIT license with its notices retained; see [NOTICE](NOTICE).

