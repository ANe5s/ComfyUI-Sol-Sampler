# ComfyUI-Sol-Sampler

[English](README.md) | 简体中文

## Sol-Attn 渐进式采样器

### 更好、更快：高分长视频一轮渐进生成

**Sol-Attn Progressive Sampler** 面向使用 ComfyUI 原生 **Model Sparse Attention（Sol-Attn）** 的高分辨率视频生成。它在一次渐进式采样执行中完成低分辨率去噪、潜空间放大和高分辨率去噪，并从同一份高分辨率噪声按所选模式构造低分辨率噪声，在两个阶段间建立跨分辨率噪声耦合；默认使用二维正交 Haar LL 噪声下采样。

## 思路来源与观察

### 从修复 SelfLift 分块上下文孤立到 Sol-Attn

这个思路的灵感来源于修复 SelfLift 采样器去噪分块造成的各分块上下文孤立问题的过程。引入分块全局上下文进行实验后，进一步发现完全取消去噪分块不仅生成更快，原本需要靠分块解决的显存问题也同时缓解了。结合节点行为、代码和后续实验，最初的问题排查最终指向 Sol-Attn；之后在 Spark-H3 上的测试，也将优化方向进一步收束到 Sol-Attn。

### 跨分辨率采样中的观察

#### 未耦合噪声场的表现

在我的测试中，低、高分阶段缺少明确的噪声耦合时——例如分别生成噪声，仅用 seed 加减建立联系——可能出现异常物体、残影和参考图异常。我观察到，两阶段跨分辨率噪声场未耦合会直接导致采样轨迹及起始点偏移，也可能引入注意力上下文混乱。例如，在 seed 相差 1 的测试中，一种结果里的远处人物未能拟合成人形，而是生成了混乱物体；另一结果中，远处人物所参考的人物错误，模型还直接复制了人物参考图，没有将其作为人物参考来生成。

可见，跨分辨率采样中高、低分辨率噪声的耦合是必须解决的问题。

#### Haar LL 的观察结果

使用二维正交 Haar LL 噪声下采样后，我观察到这些异常大幅减少，但其并非完全耦合，仍会偶发一些极小问题。

#### 双线性下采样的观察结果

使用双线性噪声下采样且不做方差补偿时，我观察到这些异常得到了解决，但超高速运动中可能出现运动拟合问题。

## 工作原理

### 错开注意力预热与潜空间增长

Sol-Attn 从稠密注意力预热逐步过渡到稀疏注意力。稠密注意力的计算和显存开销更高；注意力越稀疏，这些开销越低。高压运行时，如果稠密注意力、潜空间激活和模型权重在启动阶段同时争用资源，预热就可能变得很慢。

采样器通过错开峰值，并协调两种变化方向相反的显存需求：

```text
Sol-Attn：   稠密注意力预热 → 逐渐稀疏 → 稀疏注意力
视频潜空间：低分辨率潜空间 → 潜空间放大 → 高分辨率潜空间
噪声场：目标高分噪声 → 按模式导出低分噪声 → 切换时使用原始高分噪声
```

低分辨率阶段让 Sol-Attn 先在较小的潜空间上完成预热。当注意力上下文已预热并逐步变稀疏后，采样器再放大潜空间，并在同一轮采样中继续高分辨率去噪。这样，体积更大的潜空间会在注意力显存需求下降后进入，避免它与稠密注意力预热、模型加载的峰值重叠。

大量 ComfyUI 实际生成已验证这种方案在速度和显存方面的收益。优化机制聚焦于两个要素：Sol-Attn，以及让 Sol-Attn 视为一轮采样的低分到高分联合采样方案。它不对去噪过程进行分块，避免最初实验中发现的分块上下文孤立问题。可选的 H3 时间分块只作用于潜空间放大阶段，不会把去噪阶段拆成多个分块。

### Sigma 序列与分辨率切换

采样器在 `lowres_steps` 对应的位置切分输入 sigma 序列：先以较低空间分辨率去噪，再放大去噪后的潜空间；随后根据切换处的 sigma 重建高分辨率采样状态，并继续执行剩余 sigma。两个阶段使用同一个 ComfyUI 原生采样器；节点不会额外添加噪声序列或去噪步数。实际峰值会随模型、潜空间形状、注意力实现和其他常驻模型而变化。

### 跨分辨率噪声耦合

节点从 SamplerCustomAdvanced 接收目标高分辨率噪声，再按所选模式从同一张量导出低分辨率噪声。到切换 sigma 时，节点将放大后的低分辨率去噪潜空间与原始高分辨率噪声结合，重建高分辨率采样状态，并继续执行剩余 sigma。各噪声模式的耦合特性见[低分辨率噪声模式](#低分辨率噪声模式)。

## 低分辨率噪声模式

所选模式决定如何将目标分辨率噪声映射到低分辨率网格。

### 二维正交 Haar LL 噪声下采样



对每一帧、每个潜空间通道中的互不重叠 `2×2` 空间区块，取二维正交 Haar 变换的低低频（LL）系数：

$$
\epsilon_L[t,i,j]=\frac{\epsilon_H[t,2i,2j]+\epsilon_H[t,2i+1,2j]+\epsilon_H[t,2i,2j+1]+\epsilon_H[t,2i+1,2j+1]}{2}
$$

系数 `1/2` 是正交 Haar LL 的归一化因子。若高分噪声是独立标准高斯噪声，得到的低分噪声方差仍为 `1`。每个低分噪声值都对应一块确定的高分 `2×2` 区域：低分阶段使用二者共享的粗尺度成分，高分阶段则继续使用原始高分噪声，其中还保留了 LL 投影未包含的空间细节。此模式要求低分潜空间的宽、高都恰好是高分潜空间的一半。

#### 跨分辨率耦合

对每个高分 2×2 噪声块做正交 Haar 变换，可得到一个 LL 系数和三个细节系数。低分噪声值就是这个块的 LL 系数；高分阶段使用的仍是原始高分噪声，其中包含 LL 和三个细节系数携带的信息。因此，两种分辨率的噪声存在确定的数学关联，并非两份独立噪声。对于独立标准高斯输入，块中每个高分噪声样本与对应 LL 系数的相关系数均为 0.5。

这个共享的 LL 系数把高、低分辨率的粗尺度随机成分耦合起来，但不保证两阶段的去噪轨迹或生成视频完全相同：模型处理的空间网格不同，而且高分阶段还保留了低分投影未包含的细节成分。

### 双线性噪声下采样

对同一份高分噪声执行双线性插值。低分尺寸不是高分尺寸的一半时，可使用此模式。

#### 跨分辨率耦合

每个低分噪声值都是附近高分噪声值按双线性插值权重得到的确定性加权和，因此两阶段噪声存在确定的线性关系，而不是独立噪声场。与 Haar LL 不同，双线性插值不会把噪声分解成正交的低频系数与细节系数。高分阶段仍使用原始高分噪声，低分噪声场则经插值平滑；其方差和相关结构会受插值权重、缩放比例及采样对齐方式影响。

不做方差补偿时，双线性插值下采样会降低噪声方差。生成结果显示，较低的噪声幅度会使低分采样更接近中间去噪状态。H3 潜空间放大器接收这个中间态；由于此时潜空间中没有很多尚未正确拟合的细节，放大后可避免输出错误细节，让高分阶段获得更干净的起点，从而避免跨分辨率区块耦合带来的异常问题。但如果模型在切换点尚未充分拟合运动部分，这种路径也可能使高速运动出现异常动态模糊，表现为运动未能正确拟合。



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

## 输入

| 输入 | 默认值 | 说明 |
| --- | --- | --- |
| `sampler_name` | `euler` | 两个分辨率阶段共用的 ComfyUI 原生采样器。 |
| `lowres_steps` | `6` | 低分辨率去噪的模型评估次数；高分辨率阶段至少保留一步。 |
| `lowres_scale` | `0.5` | 低分辨率潜空间相对输入潜空间的空间比例。设为 `1.0` 时只执行一轮采样，不放大潜空间，也不进行第二阶段采样。 |
| `lowres_noise_mode` | `二维正交 Haar LL 噪声下采样` | 控制高分噪声如何映射到低分阶段。跨分辨率耦合原理见[低分辨率噪声模式](#低分辨率噪声模式)。 |
| `latent_upsample` | `Minimax H3 3D` | 选择 H3 模型放大、双线性插值或最近邻插值。 |
| `upscaler_model` | 首个可用 H3 模型；无模型时为 `none` | 从 `models/latent_upscale_models` 选择 H3 模型。若放大方式选为 `Minimax H3 3D` 但模型为 `none`，则使用双线性插值。 |
| `upscaler_precision` | `checkpoint` | H3 放大模型的推理精度：使用检查点精度、FP32、FP16 或 BF16。 |
| `upscaler_temporal_chunking` | `true` | 将较长的视频潜空间切成带重叠的时间块处理，以降低峰值工作显存；不对音频潜空间分块。 |
| `upscaler_layerwise_cpu_offload` | `false` | 将 H3 权重保留在 CPU，并在模块前向时移入 GPU、计算后移回 CPU。可降低放大阶段的峰值显存，但会增加 CPU/GPU 传输耗时。 |

如果工作流总步数少于两步，节点会退回单阶段采样，因为没有可供高分辨率阶段执行的采样步数。

## 联合优化策略

[展示工作流](examples/) 将注意力、采样器和显存管理设置组合使用。

### 注意力与显存路径

模型处理顺序：

`模型注意力后端` → `Model Sparse Attention（Sol-Attn）` → `MiniMax H3 Chunk FeedForward`

`MiniMax H3 Chunk FeedForward` 默认 `chunks=2`、`seq_threshold=4096`。如果它成为性能瓶颈，可以依次尝试将 `chunks` 提高到 `4`、`8` 或 `16`，并将 `seq_threshold` 降到 `2048`。

中高显存压力下，我建议使用 KJ/VRAMdebug 一类的显存清理节点，释放显存和内存占用。可以将它们放在潜空间传递过程中，以及最终保存视频之前。

### 放大器建议

请保持 H3 放大器的时间维度分块开启。此设置只影响潜空间放大，不会拆分去噪过程。放大表现仍可能因设备和运行环境而异。

## Turbo LoRA的选择

默认使用二维正交 Haar LL 噪声下采样后，你可以广泛使用自己喜欢的加速 LoRA。我已验证 `pika_h3_flex_turbo_8step_hd_ref2va.safetensors`：LoRA 强度 `0.55`，采样器可选 `Res_multistep`、`euler` 或 `LCM`，调度器使用 `simple`。你可以在 [Cinematic V2_Sol-Attn_Progressive_Sampler_REF2VA 工作流](examples/Cinematic%20V2_Sol-Attn_Progressive_Sampler_REF2VA.json) 中体验。
## 我的实验环境与生成记录

- 显卡：NVIDIA GeForce RTX 5070（12 GB 显存）
- CPU：AMD Ryzen 5 9600X
- 内存：32 GB

所有记录均使用 16:9 和 `lowres_scale=0.5`；`ResolutionSelector multiple` 根据工作流设置而异。表中列出各记录使用的加速 LoRA；部分生成还额外加载了风格 LoRA。FL2VA 检查点为 `minimax_h3_fl2va_pruned_int8_convrot.safetensors`；REF2VA 检查点为 `minimax_h3_ref2va_pruned_int8_convrot.safetensors`。

`采样器耗时`对应日志 `[Sol-Sampler] Sampling completed`；`完整工作流耗时`对应 ComfyUI 的 `Prompt executed`。时间单位为秒。输出文件名、尺寸、帧数和实际时长来自本地 MP4 元数据；视频文件仍保存在本地 ComfyUI 输出目录，不包含在此仓库中。

### FL2VA 模型

- **DMAD/larryvrh 混合：**`minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors`，强度 0.8–1.0；Euler / Beta；总步数 10，`lowres_steps=6`。
- **Pika：**`pika_h3_flex_fl2va_comfy_bf16.safetensors`，强度 1.0；LCM / Simple；总步数 9，`lowres_steps=5`。

| 加速 LoRA | 目标 | 采样器耗时 | 完整工作流耗时 | 备注 |
| --- | --- | ---: | ---: | --- |
| DMAD/larryvrh 混合 | 1.0 MP / 15 秒 | 375.192 秒 | 425.82 秒 | 1376 × 768，362 帧，15.083 秒 |
| DMAD/larryvrh 混合 | 1.5 MP / 10 秒 | 383.567 秒 | 443.40 秒 | 1664 × 928，243 帧，10.125 秒 |
| DMAD/larryvrh 混合 | 2.0 MP / 8 秒 | 506.743 秒 | 568.42 秒 | 1920 × 1088，192 帧，8.000 秒；`chunks=8`，`seq_threshold=2048` |
| DMAD/larryvrh 混合 | 3.0 MP / 5 秒 | 478.564 秒 | 535.06 秒 | 2368 × 1344，124 帧，5.167 秒；`chunks=8`，`seq_threshold=2048` |
| Pika | 1.0 MP / 15 秒 | 408.283 秒 | 477.36 秒 | 1360 × 768，362 帧，15.083 秒 |
| Pika | 1.5 MP / 10 秒 | 418.273 秒 | 484.01 秒 | 1680 × 944，243 帧，10.125 秒 |
| Pika | 2.0 MP / 8 秒 | 495.654 秒 | 563.76 秒 | 1936 × 1088，192 帧，8.000 秒 |
| Pika | 3.0 MP / 5 秒 | 498.364 秒 | 549.58 秒 | 2368 × 1344，124 帧，5.167 秒；`chunks=8`，`seq_threshold=2048` |

### REF2VA 模型

- **DMAD/larryvrh 混合：**`minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors`，强度 0.8–1.0；Euler / Beta；总步数 10，`lowres_steps=6`。
- **Pika：**`pika_h3_flex_ref2va_comfy_bf16.safetensors`，强度 1.0；LCM / Simple；总步数 9，`lowres_steps=5`。

| 加速 LoRA | 目标 | 采样器耗时 | 完整工作流耗时 | 备注 |
| --- | --- | ---: | ---: | --- |
| DMAD/larryvrh 混合 | 1.0 MP / 15 秒 | 510.795 秒 | 601.00 秒 | 1376 × 768，362 帧，15.083 秒 |
| DMAD/larryvrh 混合 | 1.5 MP / 10 秒 | 564.405 秒 | 667.00 秒 | 1664 × 928，243 帧，10.125 秒 |
| DMAD/larryvrh 混合 | 2.0 MP / 8 秒 | 1256.599 秒 | 1440.00 秒 | 1936 × 1088，192 帧，8.000 秒；`chunks=8`，`seq_threshold=2048` |
| DMAD/larryvrh 混合 | 3.0 MP / 5 秒 | 434.076 秒 | 500.77 秒 | 2368 × 1344，124 帧，5.167 秒 |
| Pika | 1.0 MP / 15 秒 | 560.169 秒 | 730.00 秒 | 1360 × 768，362 帧，15.083 秒 |
| Pika | 1.5 MP / 10 秒 | 426.371 秒 | 497.48 秒 | 1664 × 928，243 帧，10.125 秒 |
| Pika | 2.0 MP / 8 秒 | 486.710 秒 | 557.45 秒 | 1920 × 1088，192 帧，8.000 秒；`chunks=8`，`seq_threshold=2048` |
| Pika | 3.0 MP / 5 秒 | 736.431 秒 | 803.00 秒 | 2368 × 1344，124 帧，5.167 秒；`chunks=8`，`seq_threshold=2048` |

在我更广泛的本地生成记录中，T2VA 和 REF2VA 的耗时表现总体相近。本表中的严格对照尚不完整：目前只完成了一组 REF2VA / Pika 对照，而且本次测量时还有其他桌面程序运行。剩余耗时将在后台负载较低时直接运行后补充。
## 放大模型的内存行为

H3 放大模型会缓存在 CPU 内存中供后续复用。标准模型管理模式下，潜空间转换完成后，节点会请求 ComfyUI 模型管理器将放大模型权重从受管理的 GPU 显存卸载。启用逐层 CPU 卸载时，则在每个模块前向计算后将其移回 CPU。两种模式下放大模型权重都会保留在系统内存缓存中；放大后的潜空间会继续保留，供高分辨率采样使用。

## 致谢

- **[H3-潜在升频器](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)** — 本项目的 `h3_latent_upscaler.py` 改编自 LBH-123-AI 的节点实现，并保留了上游 MIT 许可证声明，详见 [NOTICE](NOTICE)。
- **[Sol-H3](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/)** 与 **[Sol-Engine](https://nvlabs.github.io/Sana/Sol-Engine/)** — NVIDIA Research 的 H3 推理与系统加速相关项目，作为相关研究与工程参考。
- **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)** — 感谢 Kijai 提供的 ComfyUI 辅助节点；展示工作流使用了其中的显存与系统内存清理工具。
- **[Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/)** — 本项目工作流使用 ComfyUI 原生 Model Sparse Attention（Sol-Attn）。
- **[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)** — 本项目面向的基础视频与音频生成模型。
- **[larryvrh / MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)** — 本项目的 Turbo LoRA 测试与 DMAD/larryvrh 混合工作流记录使用了该模型系列。
- **[DMAD：Distribution Matching as Adversarial Distillation for Fast Visual Generation](https://huggingface.co/ZhengmingYu/DMAD)** — 感谢 Zhengming Yu 等作者提出 DMAD 方法并开源 MiniMax H3 DMAD LoRA；本项目测试使用的混合 LoRA 包含该适配器。
- **[tsolful / DMAD + larryvrh v4 混合 LoRA](https://civitai.red/models/2837571/minimax-h3-turbo-loras?modelVersionId=3391964)** — 感谢 tsolful 发布该混合 LoRA，本项目的二次采样测试中使用了它。
- **[Pika H3 Flex](https://github.com/Pika-Labs/Pika-H3-Flex)** — 展示工作流与生成测试中使用的开放加速模型与适配器。
- **[MiniMax H3 Cinematic V2](https://civitai.red/models/2890588/minimax-h3-cinematic-v2?modelVersionId=3417955)** — 感谢这款风格 LoRA 的作者；它用于 REF2VA 展示工作流。

## 许可证

项目自有代码与文档采用 Apache License 2.0，详见 [LICENSE](LICENSE)。`h3_latent_upscaler.py` 文件包含改编自上游的代码，按上游 MIT 许可证分发并保留其声明，详见 [NOTICE](NOTICE)。

