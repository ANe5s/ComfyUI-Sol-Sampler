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

