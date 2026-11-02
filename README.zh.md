# ComfyUI-Sol-Sampler（简体中文）

[English](README.md) | 简体中文

## Sol-Attn 渐进式采样器：闪电般的速度

**Sol-Attn Progressive Sampler** 面向使用 ComfyUI 原生 **Model Sparse Attention（Sol-Attn）** 的高分辨率视频生成。它在一次渐进式采样执行中完成低分辨率去噪、潜空间放大和高分辨率去噪。

## 这个思路的由来

这个思路起源于 SelfLift 采样器中观察到的问题：去噪分块会造成各分块上下文孤立。引入分块全局上下文进行实验后，进一步发现完全取消去噪分块不仅生成更快，原本需要靠分块解决的显存问题也同时缓解了。结合节点行为、代码和后续实验，问题最终指向 Sol-Attn；之后又在 Spark-H3 上测试了这一方法。

## 工作原理

Sol-Attn 从稠密注意力预热逐步过渡到稀疏注意力。稠密注意力的计算和显存开销更高；注意力越稀疏，这些开销越低。高压运行时，如果稠密注意力、潜空间激活和模型权重在启动阶段同时争用资源，预热就可能变得很慢。

采样器通过错开峰值，并协调两种变化方向相反的显存需求：

```text
Sol-Attn：   稠密注意力预热 → 逐渐稀疏 → 稀疏注意力
视频潜空间：低分辨率潜空间 → 潜空间放大 → 高分辨率潜空间
```

低分辨率阶段让 Sol-Attn 先在较小的潜空间上完成预热。当注意力上下文已预热并逐步变稀疏后，采样器再放大潜空间，并在同一轮采样中继续高分辨率去噪。这样，体积更大的潜空间会在注意力显存需求下降后进入，避免它与稠密注意力预热、模型加载的峰值重叠。

大量 ComfyUI 实际生成已验证这种方案在速度和显存方面的收益。优化机制聚焦于两个要素：Sol-Attn，以及让 Sol-Attn 视为一轮采样的低分到高分联合采样方案。它不对去噪过程进行分块，避免最初实验中发现的分块上下文孤立问题。可选的 H3 时间分块只作用于潜空间放大阶段，不会把去噪阶段拆成多个分块。

采样器在 `lowres_steps` 对应的位置切分输入 sigma 序列：先以较低空间分辨率去噪，再放大去噪后的潜空间；随后根据切换处的 sigma 重建高分辨率采样状态，并继续执行剩余 sigma。两个阶段使用同一个 ComfyUI 原生采样器；节点不会额外添加噪声序列或去噪步数。实际峰值会随模型、潜空间形状、注意力实现和其他常驻模型而变化。

## 我的实验环境与生成记录

- 显卡：NVIDIA GeForce RTX 5070（12 GB 显存）
- CPU：AMD Ryzen 5 9600X
- 内存：32 GB

所有记录均使用 16:9、`ResolutionSelector multiple=16`、`lowres_scale=0.5`，且只启用表中列出的加速 LoRA，不额外加载风格 LoRA。FL2VA 检查点为 `minimax_h3_fl2va_pruned_int8_convrot.safetensors`；REF2VA 检查点为 `minimax_h3_ref2va_pruned_int8_convrot.safetensors`。

`采样器耗时`对应日志 `[Sol-Sampler] Sampling completed`；`完整工作流耗时`对应 ComfyUI 的 `Prompt executed`。时间单位为秒。输出文件名、尺寸、帧数和实际时长来自本地 MP4 元数据；视频文件仍保存在本地 ComfyUI 输出目录，不包含在此仓库中。

### FL2VA 模型

- **DMAD/larryvrh 混合：**`minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors`，强度 0.8–1.0；Euler / Beta；总步数 10，`lowres_steps=6`。
- **Pika：**`pika_h3_flex_fl2va_comfy_bf16.safetensors`，强度 1.0；LCM / Simple；总步数 9，`lowres_steps=5`。

| 加速 LoRA | 目标 | 采样器耗时 | 完整工作流耗时 | 备注 |
| --- | --- | ---: | ---: | --- |
| DMAD/larryvrh 混合 | 1.0 MP / 15 秒 | — | — | 未测；待低后台负载时直接运行。 |
| DMAD/larryvrh 混合 | 1.5 MP / 10 秒 | — | — | 未测；待低后台负载时直接运行。 |
| DMAD/larryvrh 混合 | 2.0 MP / 8 秒 | — | — | 未测；待低后台负载时直接运行。 |
| DMAD/larryvrh 混合 | 3.0 MP / 5 秒 | — | — | 未测；待低后台负载时直接运行。 |
| Pika | 1.0 MP / 15 秒 | 408.283 秒 | 477.36 秒 | 1360 × 768，362 帧，15.083 秒 |
| Pika | 1.5 MP / 10 秒 | 418.273 秒 | 484.01 秒 | 1680 × 944，243 帧，10.125 秒 |
| Pika | 2.0 MP / 8 秒 | 495.654 秒 | 563.76 秒 | 1936 × 1088，192 帧，8.000 秒 |
| Pika | 3.0 MP / 5 秒 | — | — | H3 潜空间放大在时间分块 32→16→8 后仍 OOM；未生成完成视频。 |

### REF2VA 模型

- **检查点：**`minimax_h3_ref2va_pruned_int8_convrot.safetensors`。
- **DMAD/larryvrh 混合：**`minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors`，强度 0.8–1.0；Euler / Beta；总步数 10，`lowres_steps=6`。
- **Pika：**`pika_h3_flex_ref2va_comfy_bf16.safetensors`，强度 1.0；LCM / Simple；总步数 9，`lowres_steps=5`。

| 加速 LoRA | 目标 | 采样器耗时 | 完整工作流耗时 | 备注 |
| --- | --- | ---: | ---: | --- |
| DMAD/larryvrh 混合 | 1.0 MP / 15 秒 | — | — | 未测；待低后台负载时直接运行。 |
| DMAD/larryvrh 混合 | 1.5 MP / 10 秒 | — | — | 未测；待低后台负载时直接运行。 |
| DMAD/larryvrh 混合 | 2.0 MP / 8 秒 | — | — | 1936 × 1088，192 帧，8.000 秒 |
| DMAD/larryvrh 混合 | 3.0 MP / 5 秒 | — | — | 2368 × 1344，124 帧，5.167 秒 |
| Pika | 1.0 MP / 15 秒 | 560.169 秒 | 730.00 秒 | 1360 × 768，362 帧，15.083 秒 |
| Pika | 1.5 MP / 10 秒 | — | — | 按用户要求在完成前停止；待直接运行。 |
| Pika | 2.0 MP / 8 秒 | — | — | 未测；待直接运行。 |
| Pika | 3.0 MP / 5 秒 | — | — | 未测；待直接运行。 |

在我更广泛的本地生成记录中，T2VA 和 REF2VA 的耗时表现总体相近。本表中的严格对照尚不完整：目前只完成了一组 REF2VA / Pika 对照，而且本次测量时还有其他桌面程序运行。剩余耗时将在后台负载较低时直接运行后补充。
## 使用要求

- ComfyUI 中可用原生 **Model Sparse Attention（Sol-Attn）** 节点。
- 使用兼容的图像或视频潜空间工作流。空间视频潜空间会放大；音频等非空间流保持原有形状。
- 使用模型放大时，需要 MiniMax H3 3D 潜空间放大模型（`.safetensors` 或 `.pth`）。

## 安装

将此文件夹放入 `ComfyUI/custom_nodes/ComfyUI-Sol-Sampler`，然后重启 ComfyUI。

使用 **Minimax H3 3D** 放大时，将模型放入：

```text
ComfyUI/models/latent_upscale_models/
```

添加模型后重启 ComfyUI，使其出现在节点的模型列表中。

## 工作流连接

1. 使用 ComfyUI 原生 **Model Sparse Attention（Sol-Attn）** 处理扩散模型。
2. 将处理后的模型连接到引导器（例如 `CFGGuider`）的 `MODEL` 输入。
3. 将引导器连接到 `SamplerCustomAdvanced`。
4. 将 **Sol-Attn Progressive Sampler** 连接到 `SamplerCustomAdvanced` 的 `sampler` 输入。
5. 按常规方式连接噪声、sigma 序列和潜空间输入。

节点位于 **ANe5s Nodes → Sol-Sampler**，输出类型为 `SAMPLER`。

## 联合优化策略

[展示工作流](examples/) 将注意力、采样器和显存管理设置组合使用。

### 注意力与显存路径

模型处理顺序：

`模型注意力后端` → `Model Sparse Attention（Sol-Attn）` → `MiniMax H3 Chunk FeedForward`

`MiniMax H3 Chunk FeedForward` 默认 `chunks=2`、`seq_threshold=4096`。如果它成为性能瓶颈，可以依次尝试将 `chunks` 提高到 `4`、`8` 或 `16`，并将 `seq_threshold` 降到 `2048`。

中高显存压力下，我建议使用 KJ/VRAMdebug 一类的显存清理节点，释放显存和内存占用。可以将它们放在潜空间传递过程中，以及最终保存视频之前。

### 采样器路径

在模型上应用 ComfyUI 原生 `Model Sparse Attention（Sol-Attn）`，再将 `Sol-Attn Progressive Sampler` 接到 `SamplerCustomAdvanced` 的 `sampler` 输入。Sol-Attn 通过逐渐稀疏化注意力降低注意力侧的显存压力；渐进式采样器则在一轮采样中协调低分辨率去噪、潜空间放大和高分辨率去噪。它实现了分块去噪通常用于缓解显存压力的目标，同时保留共享采样上下文。

请保持 `upscaler_temporal_chunking` 开启（默认）。时间分块只用于潜空间放大。当前实现优先兼顾更多设备环境，放大阶段仍可能因设备和运行环境差异而偶发问题。

## 展示工作流为何选用 Pika H3 Flex

这部分讨论以 2026 年 8 月 7 日及之后我公开的相关研究为背景。我在 larryvrh V1 step500 版 Turbo LoRA 中观察到一种与模型或训练方法有关的特性：在 larryvrh Turbo LoRA 的早期训练阶段，其加速路径尚未出现明显过拟合。将基模与 LoRA 组合，并沿完整 Sigma 轨迹去噪时，可以明显观察到：前段归一化 Sigma 高值区间与后段归一化 Sigma 低值区间的去噪方向和特征明显不同。高 Sigma 阶段主要影响整体结构，低 Sigma 阶段主要影响细节。由此可以清楚地看到，单一 Turbo LoRA 在不同归一化 Sigma 区间仍呈现出高、低噪阶段各自的特征。

但在后续加速模型中，这些特征消失了。larryvrh、lightx2v 等后续版本确实学会了用更少步数逼近目标去噪轨迹，但在后续加速模型中，这种高低噪特性消失了。以标称 4／8 步的加速 LoRA 为例，即使增加采样步数、延长低 Sigma 阶段，新增的去噪步也未能继续改善画面；油腻、AI 涂抹感仍与 8 步结果相近，仿佛画质上限被加速路径锁定。

Civitai 作者 tsolful 将 DMAD 与 larryvrh 的 v4_step600 合并后，这一问题得到了解决，LoRA 加速路径的过拟合也有所缓解。经过 B 站作者 YZ_金鱼的广泛测试与工作流分享，`minimax_h3_dmad_full_dareties_v4_step600_comfy` 凭借更好的表现，逐渐被广泛用于二次采样。不过，这种权重合并也会带来代价：某些物体（例如机械）可能出现形体欠拟合、结构崩坏等问题。tsolful 的模型页面

Pika H3 Flex 开源后，它的六步加速 LoRA 呈现出我熟悉的加速路径欠拟合特征。我重新引入并测试了这款 LoRA，也观察到了熟悉的高低噪行为。经过实验，我可以初步得出以下结论：LCM 采样器的更新规则相对固定，但实际去噪路径仍受 Sigma 序列影响。在常见的二次采样流程中，直接沿用官方推荐的固定六步 Sigma 序列会出现不少问题；我的测试表明，实际设置为九步以上更合适，使用调度器生成 Sigma 时优先选择 Simple，其次是 Beta。低分辨率阶段设置五步是必要的：如果这五步使用的 Sigma 序列未覆盖归一化后的高 Sigma 区间，容易导致某些物体完全缺失；如果分配给低 Sigma 区间的步数不足，快速运动物体则容易欠拟合。

### Pika 采样参数

| 参数 | 设置 |
| --- | --- |
| 采样器 | FL2VA 和 REF2VA 均使用 `lcm`。 |
| 调度器 | 优先 `Simple`，其次 `Beta`。 |
| 总步数 | 九步以上；展示工作流使用九步。 |
| `lowres_steps` | `5`；总步数为九步时，高分辨率阶段为四步。 |
## 输入

| 输入 | 默认值 | 说明 |
| --- | --- | --- |
| `sampler_name` | `euler` | 两个分辨率阶段共用的 ComfyUI 原生采样器。 |
| `lowres_steps` | `6` | 低分辨率去噪的模型评估次数；高分辨率阶段至少保留一步。 |
| `lowres_scale` | `0.5` | 低分辨率潜空间相对输入潜空间的空间比例。设为 `1.0` 时只执行一轮采样，不放大潜空间，也不进行第二阶段采样。 |
| `latent_upsample` | `Minimax H3 3D` | 选择 H3 模型放大、双线性插值或最近邻插值。 |
| `upscaler_model` | 首个可用 H3 模型；无模型时为 `none` | 从 `models/latent_upscale_models` 选择 H3 模型。若放大方式选为 `Minimax H3 3D` 但模型为 `none`，则使用双线性插值。 |
| `upscaler_precision` | `checkpoint` | H3 放大模型的推理精度：使用检查点精度、FP32、FP16 或 BF16。 |
| `upscaler_temporal_chunking` | `true` | 将较长的视频潜空间切成带重叠的时间块处理，以降低峰值工作显存；不对音频潜空间分块。 |
| `upscaler_layerwise_cpu_offload` | `false` | 将 H3 权重保留在 CPU，并在模块前向时移入 GPU、计算后移回 CPU。可降低放大阶段的峰值显存，但会增加 CPU/GPU 传输耗时。 |

当 `lowres_scale` 为 `1.0` 时，放大器相关设置不会使用。如果工作流总步数少于两步，也会退回单阶段采样，因为没有可供高分辨率阶段执行的采样步数。

## 放大模型的内存行为

H3 放大模型会缓存在 CPU 内存中供后续复用。标准模型管理模式下，潜空间转换完成后，节点会请求 ComfyUI 模型管理器将放大模型权重从受管理的 GPU 显存卸载。启用逐层 CPU 卸载时，则在每个模块前向计算后将其移回 CPU。两种模式下放大模型权重都会保留在系统内存缓存中；放大后的潜空间会继续保留，供高分辨率采样使用。

## 许可证

项目自有代码与文档采用 Apache License 2.0，详见 [LICENSE](LICENSE)。`h3_latent_upscaler.py` 文件包含改编自上游的代码，按上游 MIT 许可证分发并保留其声明，详见 [NOTICE](NOTICE)。
