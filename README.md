# ⚡ ComfyUI-Sol-Sampler

[简体中文](README.zh.md)

## Sol-Attn Progressive Sampler

### Better, Faster: One-Pass Progressive Generation for High-Resolution, Long-Form Video

**Sol-Attn Progressive Sampler** is a sampling strategy for high-resolution video generation with ComfyUI's native **Model Sparse Attention (Sol-Attn)**. It performs low-resolution denoising, latent upscaling, and high-resolution denoising in one progressive sampling execution. It derives low-resolution noise from the same high-resolution noise field using the selected mode, coupling noise across the two stages; 2D orthonormal Haar LL noise downsampling is the default.

## Motivation and observations

### From fixing SelfLift chunk-context isolation to Sol-Attn

This idea was inspired by the work of fixing context isolation between chunks caused by denoising chunking in the SelfLift sampler. Experiments with global context showed that removing denoising chunks made generation faster, while also resolving the memory problem that chunking was meant to address. Node behavior, code inspection, and further experiments first pointed to Sol-Attn; subsequent testing with Spark-H3 further narrowed the optimization direction to Sol-Attn.

### Observations in cross-resolution sampling

#### Effects of uncoupled noise fields

In my tests, when the low- and high-resolution stages lacked explicit noise coupling—for example, when their noise fields were generated separately and related only by adding or subtracting seed values—anomalous objects, ghosting, and reference-image errors could occur. I observed that uncoupled cross-resolution noise fields between the two stages directly shift the sampling trajectory and its starting point, and may also introduce attention-context confusion. For example, in tests where the noise seeds differed by one, the distant person in one result failed to resolve into a person and instead became a malformed object; in another, the distant figure was associated with the wrong reference character, and the model copied the character-reference image directly instead of using it as a character reference.

This shows that coupling noise between the low- and high-resolution stages is an issue that must be addressed in cross-resolution sampling.

#### Haar LL result

I observed that 2D orthonormal Haar LL noise downsampling greatly reduced these anomalies, but the coupling is not complete and a few very minor issues can still occur occasionally.

#### Bilinear result

With bilinear noise downsampling without variance compensation, I observed that these anomalies were resolved, though ultra-fast motion may introduce motion-fitting issues.

## How it works

### Stagger attention warm-up and latent growth

Sol-Attn moves from dense attention during warm-up toward sparse attention. Dense attention has higher compute and memory costs; as attention becomes sparser, those costs fall. Under high memory pressure, the initial warm-up can be slow when dense attention, latent activations, and model weights all compete for resources at once.

The sampler separates that peak in time and coordinates two changing memory costs:

```text
Sol-Attn:     dense warm-up  →  progressively sparse  →  sparse attention
Video latent: low-resolution latent → latent upscale → high-resolution latent
Noise field: high-resolution source → mode-derived low-resolution noise → original high-resolution noise at transition
```

The low-resolution phase lets Sol-Attn warm up while the latent is small. After the attention context is warmed and sparsity has increased, the sampler upscales the latent and continues denoising at high resolution in the same sampling run. The larger latent therefore enters when attention requires less memory, instead of overlapping its peak with dense-attention warm-up and model loading.

Extensive ComfyUI generation runs have validated the speed and memory benefits of this approach in practice. The optimization is built around two elements: Sol-Attn and a joint low-/high-resolution sampling plan that Sol-Attn treats as one sampling run. It avoids chunked denoising, removing the context isolation that motivated the original investigation. The optional H3 temporal chunking setting applies only inside latent upscaling; it does not split the denoising stages.

### Sigma schedule and resolution transition

The sampler splits the input sigma schedule at `lowres_steps`. It denoises the latent at reduced spatial resolution, upscales the denoised latent, reconstructs the high-resolution state at the transition sigma, then continues with the remaining sigmas. The same selected native ComfyUI sampler is used in both stages; the node does not add a second noise schedule or extra denoising steps. Actual peak usage varies with the model, latent shape, attention implementation, and other models resident in memory.

### Cross-resolution noise coupling

The node receives target-resolution noise from SamplerCustomAdvanced, then derives low-resolution noise from that same tensor using the selected mode. At the transition sigma, it combines the upscaled low-resolution denoised latent with the original high-resolution noise to reconstruct the high-resolution state, then continues the remaining schedule. The mode-specific coupling properties are described in [Low-resolution noise modes](#low-resolution-noise-modes).

## Low-resolution noise modes

The selected mode defines how target-resolution noise is mapped onto the low-resolution grid.

### 2D orthonormal Haar LL noise downsampling



Uses the low-low (LL) coefficient of each non-overlapping 2×2 spatial block, independently for every frame and latent channel:

$$
\epsilon_L[t,i,j]=\frac{\epsilon_H[t,2i,2j]+\epsilon_H[t,2i+1,2j]+\epsilon_H[t,2i,2j+1]+\epsilon_H[t,2i+1,2j+1]}{2}
$$

The factor of 1/2 is the normalization for the orthonormal Haar LL coefficient. If the high-resolution samples are iid standard Gaussian, the resulting low-resolution samples also have unit variance. Each low-resolution value is tied to the exact 2×2 high-resolution neighborhood. The low-resolution branch uses this shared coarse component, while the high-resolution branch retains the original noise, including spatial detail omitted by the LL projection. This mode requires both spatial dimensions to be exactly halved.

#### Cross-resolution coupling

For each high-resolution 2×2 noise block, the orthonormal Haar transform yields one LL coefficient and three detail coefficients. The low-resolution noise value is exactly that block's LL coefficient; the original high-resolution noise still contains information from the LL and all three detail coefficients. The two resolutions therefore have a deterministic mathematical relationship rather than independent noise fields. For iid standard-Gaussian input, each high-resolution sample in a block has correlation 0.5 with that block's LL coefficient.

This shared LL component couples the coarse-scale random content across resolutions. It does not make the denoising trajectories or generated videos identical: the model processes different spatial grids, and the high-resolution stage retains detail components omitted by the low-resolution projection.

### Bilinear noise downsampling

Resizes the same high-resolution noise with bilinear interpolation. Use it when the low-resolution dimensions are not exactly half of the high-resolution dimensions.

#### Cross-resolution coupling

Each low-resolution noise value is a deterministic weighted combination of nearby high-resolution samples. The two stages therefore have a defined linear relationship rather than independent noise fields. Unlike Haar LL, bilinear interpolation does not separate the noise into orthogonal low-frequency and detail coefficients. The high-resolution stage retains the original noise, while the low-resolution field is smoothed; its variance and correlation pattern depend on the interpolation weights, resize ratio, and sampling alignment.

Without variance compensation, bilinear downsampling reduces the noise variance. Generation results show that the lower noise amplitude brings low-resolution sampling closer to an intermediate denoising state. The H3 latent upscaler receives this intermediate state; because the latent does not yet contain many details that have not been fitted correctly, upscaling can avoid outputting incorrect details and give the high-resolution stage a cleaner starting point, thereby avoiding anomalies associated with cross-resolution block coupling. However, if the model has not sufficiently fitted motion by the transition point, this path may cause abnormal blur during high-speed movement, indicating that the motion was not fitted correctly.



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

## Inputs

| Input | Default | Description |
| --- | --- | --- |
| `sampler_name` | `euler` | Native ComfyUI sampler used by both resolution stages. |
| `lowres_steps` | `6` | Number of denoiser evaluations at low resolution. At least one step is reserved for the high-resolution stage. |
| `lowres_scale` | `0.5` | Spatial scale for the low-resolution latent, relative to the input latent. Set to `1.0` to run a single sampling pass without latent upscaling or a second stage. |
| `lowres_noise_mode` | `2D orthonormal Haar LL noise downsampling` | Maps high-resolution noise to the low-resolution stage. See [Low-resolution noise modes](#low-resolution-noise-modes) for the cross-resolution coupling behavior. |
| `latent_upsample` | `Minimax H3 3D` | Use the learned H3 upscaler, bilinear interpolation, or nearest-neighbor interpolation. |
| `upscaler_model` | First available H3 checkpoint, or `none` | H3 checkpoint in `models/latent_upscale_models`. If `Minimax H3 3D` is selected with `none`, bilinear interpolation is used. |
| `upscaler_precision` | `checkpoint` | Inference dtype for the H3 upscaler: checkpoint dtype, FP32, FP16, or BF16. |
| `upscaler_temporal_chunking` | `true` | Processes long video latents in overlapping temporal chunks to reduce peak working memory. Audio latents are not chunked. |
| `upscaler_layerwise_cpu_offload` | `false` | Keeps H3 weights on CPU and moves modules to the GPU for their forward pass. This reduces the upscaler's peak VRAM use at the cost of CPU/GPU transfer time. |

If the workflow has fewer than two sampling steps, the node falls back to a single pass because there would be no high-resolution step to run.

## Joint optimization strategy

The [showcase workflows](examples/) combine attention, sampling, and memory-management optimizations.

### Attention and memory path

Apply model processing in this order:

`Model Attention Backend` → `Model Sparse Attention (Sol-Attn)` → `MiniMax H3 Chunk FeedForward`

`MiniMax H3 Chunk FeedForward` defaults to `chunks=2` and `seq_threshold=4096`. If it becomes a performance bottleneck, try increasing `chunks` to `4`, `8`, or `16`, and lowering `seq_threshold` to `2048`.

For medium-to-high memory pressure, I recommend KJ/VRAMdebug cleanup nodes to release VRAM and host-memory usage. Place them at latent handoffs and before the final video save.

### Upscaler recommendation

Keep temporal chunking enabled for the H3 upscaler. This setting affects latent upscaling only, not denoising. Upscaling may still behave differently across devices and runtime environments.

## Why the showcase workflows use Pika H3 Flex

This discussion is framed by the related research I made public on and after August 7, 2026. My investigation of accelerated LoRA behavior in second-sampling workflows was inspired by the [larryvrh V1 step500 Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora). In its early training stage, I observed a characteristic that may be related to the model or training method: its acceleration path had not yet shown obvious overfitting. When combining the base model with the LoRA and denoising along the full sigma trajectory, the denoising directions and characteristics in the early high normalized-sigma range and the later low normalized-sigma range are clearly different. The high-sigma stage mainly affects global structure, while the low-sigma stage mainly affects detail. This shows that one Turbo LoRA can retain distinct high- and low-noise characteristics across normalized sigma ranges.

Later larryvrh and LightX2V versions learned to approximate the target denoising trajectory in fewer steps, but no longer exhibited this high-/low-noise behavior. For accelerated LoRAs labeled for 4 or 8 steps, even increasing the sampling steps and extending the low-sigma phase did not make the additional denoising steps improve the image. The oily, AI-smeared appearance remained similar to the 8-step result, as if the acceleration path had locked the image-quality ceiling.

After Civitai creator tsolful merged DMAD with larryvrh's v4_step600, this issue was resolved and overfitting in the LoRA acceleration path was alleviated. I also included this DMAD/larryvrh merge in my second-sampling tests. Following extensive testing and workflow sharing by Bilibili creator YZ_金鱼, `minimax_h3_dmad_full_dareties_v4_step600_comfy` gradually became widely used for second sampling because of its better results. This weight merge also has a tradeoff: some objects, such as mechanical forms, may become underfit in shape and suffer structural breakdown. See [tsolful's model page](https://civitai.red/models/2837571/minimax-h3-turbo-loras?modelVersionId=3391964).

After Pika H3 Flex was released, its six-step acceleration LoRA showed the acceleration-path underfitting I was familiar with. I brought this LoRA back into my tests and observed the familiar high-/low-noise behavior. My experiments suggest the following: the LCM sampler's update rule is relatively fixed, but the actual denoising path is still affected by the sigma sequence. In common second-sampling workflows, directly reusing the officially recommended fixed six-step sigma schedule causes a number of problems. My tests indicate that nine or more steps are more suitable. When generating sigmas with a scheduler, prefer Simple, followed by Beta. Five low-resolution steps are necessary: if their sigma sequence does not cover the normalized high-sigma range, some objects may disappear completely; if too few steps are assigned to the low-sigma range, fast-moving objects may be underfit.

### Pika sampling settings

| Setting | Selection |
| --- | --- |
| Sampler | `lcm` for both FL2VA and REF2VA. |
| Scheduler | `Simple` first, followed by `Beta`. |
| Total steps | Nine or more; the showcase workflows use nine. |
| `lowres_steps` | `5`; with nine total steps, the high-resolution stage has four steps. |
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
## Upscaler memory behavior

The H3 upscaler is cached in CPU memory for reuse. In standard model-management mode, after the latent transition the node asks ComfyUI's model manager to unload the upscaler weights from managed GPU memory. With layerwise CPU offload enabled, each module is returned to CPU after its forward pass instead. In both modes the upscaler weights remain cached in host RAM; the upscaled latent remains available for the high-resolution sampling stage.

## Acknowledgements

- **[H3 Latent Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)** — This project's `h3_latent_upscaler.py` adapts LBH-123-AI's node implementation and retains its upstream MIT notice; see [NOTICE](NOTICE).
- **[Sol-H3](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/)** and **[Sol-Engine](https://nvlabs.github.io/Sana/Sol-Engine/)** — Related NVIDIA Research projects on H3 inference and system acceleration, cited as related research and engineering references.
- **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)** — Thanks to Kijai for the utility nodes used in showcase workflows for VRAM and host-memory cleanup.
- **[Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/)** — The workflows use ComfyUI's native Model Sparse Attention (Sol-Attn).
- **[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)** — The base video-and-audio model targeted by this project.
- **[larryvrh / MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)** — Used in the project's Turbo LoRA tests and DMAD/larryvrh merge workflow records.
- **[DMAD: Distribution Matching as Adversarial Distillation for Fast Visual Generation](https://huggingface.co/ZhengmingYu/DMAD)** — Credit to Zhengming Yu et al. for the DMAD method and original MiniMax-H3 DMAD LoRAs, incorporated into the merged LoRA used in this project's tests.
- **[tsolful / DMAD + larryvrh v4 merged LoRA](https://civitai.red/models/2837571/minimax-h3-turbo-loras?modelVersionId=3391964)** — Thank you to tsolful for publishing the merged LoRA used in the project's second-sampling tests.
- **[Pika H3 Flex](https://github.com/Pika-Labs/Pika-H3-Flex)** — The open acceleration model and adapters used in showcase workflows and generation tests.

## License

Original project code and documentation are licensed under the Apache License 2.0; see [LICENSE](LICENSE). The `h3_latent_upscaler.py` file, which contains code adapted from upstream, is distributed under the upstream MIT license with its notices retained; see [NOTICE](NOTICE).


