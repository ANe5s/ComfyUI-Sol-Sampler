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

I use Pika H3 Flex as the baseline acceleration LoRA in the showcase workflows. This discussion is framed by related research I made public on and after August 7, 2026.

With the larryvrh V1 step500 Turbo LoRA, I observed that its early-training acceleration path had not yet visibly overfit. Running the base model with the LoRA through a full sigma trajectory showed distinct effects by normalized sigma: high-sigma steps mainly shaped global structure, while low-sigma steps mainly refined detail. The acceleration path therefore retained recognizable high- and low-noise characteristics.

In some later accelerated LoRAs from larryvrh and LightX2V, I found that this distinction weakened or disappeared. Although these models approximate a target trajectory in fewer steps, adding steps or extending the low-sigma tail often did not improve my results: an oily or smeared appearance remained close to the nominal 8-step output, as if the acceleration path imposed a quality ceiling.

In my tests, Civitai creator tsolful's merge of DMAD with larryvrh v4_step600 alleviated this overfitting behavior. Extensive testing and workflow sharing by Bilibili creator YZ_金鱼 also helped `minimax_h3_dmad_full_dareties_v4_step600_comfy` become a common second-sampling choice. The tradeoff I observed is shape underfitting on some objects—mechanical forms in particular—which can damage structure.

Pika H3 Flex's open six-step acceleration LoRA showed a similar underfitting signature, but retesting it also revealed the high-/low-sigma behavior I was looking for. That responsiveness is why I chose Pika as the baseline for these workflows: it leaves room to improve results through sigma scheduling and low-/high-resolution step allocation.

Although an LCM sampler uses a fixed update rule, the denoising trajectory still depends on its sigma schedule. In my second-sampling tests, the fixed six-step sigma schedule caused many issues. The experimental conclusion is to use at least 9 total steps. I prefer the Simple scheduler, followed by Beta. For the showcased Pika setup, set `lowres_steps=5`; those steps should cover the normalized high-sigma region, or some objects may disappear. Keep enough steps for the low-sigma region as well, or fast-moving objects can be underfit.

### Pika settings in the showcase workflows

These Pika-specific settings differ from the node defaults listed below.

| Setting | Recommendation |
| --- | --- |
| Sampler | FL2VA and REF2VA: `lcm`. |
| Scheduler | `simple` first; `beta` as the alternative from my tests. |
| Total steps | At least `9`; the showcase workflows use `9`. This is my experimental conclusion for Pika second sampling. |
| `lowres_steps` | `5`; the remaining steps run in the high-resolution stage. At 9 total steps, that leaves 4 high-resolution steps. |

## Inputs

| Input | Default | Description |
| --- | --- | --- |
| `sampler_name` | `euler` | Native ComfyUI sampler used by both resolution stages. |
| `lowres_steps` | `6` | Number of denoiser evaluations at low resolution. At least one step is reserved for the high-resolution stage. |
| `lowres_scale` | `0.5` | Spatial scale for the low-resolution latent, relative to the input latent. Set to `1.0` to run a single sampling pass without latent upscaling or a second stage. |
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
