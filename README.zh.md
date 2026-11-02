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

我在展示工作流中选用 Pika H3 Flex 作为基准加速 LoRA。这部分讨论以 2026 年 8 月 7 日及之后我公开的相关研究为背景。

使用 larryvrh V1 step500 Turbo LoRA 时，我观察到它在早期训练阶段的加速路径尚未明显过拟合。将基模与 LoRA 组合，并沿完整 Sigma 轨迹去噪，可以看到不同归一化 Sigma 区间的作用不同：高 Sigma 阶段主要塑造整体结构，低 Sigma 阶段主要细化细节。这说明该加速路径仍保留了可辨认的高、低噪声阶段特征。

在 larryvrh 和 LightX2V 后续的一些加速 LoRA 中，我观察到这种区分逐渐减弱或消失。它们学会了用更少步数逼近目标去噪轨迹，但在我的测试中，增加步数或延长低 Sigma 尾段往往没有继续改善画面：油腻感或 AI 涂抹感仍接近标称 8 步的结果，仿佛加速路径锁定了画质上限。

在我的测试中，Civitai 作者 tsolful 将 DMAD 与 larryvrh v4_step600 合并后，缓解了这类过拟合。经过 B 站作者 YZ_金鱼的广泛测试与工作流分享，`minimax_h3_dmad_full_dareties_v4_step600_comfy` 也逐渐成为常见的二次采样选择。我观察到的代价是部分物体可能出现形体欠拟合，尤其是机械结构，进而导致结构损坏。

Pika H3 Flex 开源的六步加速 LoRA 呈现出类似的加速路径欠拟合特征；重新测试后，我也观察到了熟悉的高、低 Sigma 行为。因此我选择它作为展示工作流的基准：它对 Sigma 序列仍有响应，可以通过调度 Sigma 以及分配低、高分辨率阶段步数来继续改善结果。

LCM 采样器的更新规则相对固定，但实际去噪轨迹仍受 Sigma 序列影响。我的二次采样实验中，固定六步 Sigma 序列会出现不少问题。实验结论是总步数应设为 9 步以上。调度器优先选择 Simple，其次是 Beta。展示工作流中的 Pika 配置将 `lowres_steps` 设为 `5`；这五步的 Sigma 序列应覆盖归一化后的高 Sigma 区间，否则部分物体可能完全缺失；低 Sigma 区间也要保留足够步数，否则快速运动物体容易欠拟合。

### 展示工作流中的 Pika 参数

这些是 Pika 专用设置，与下方节点通用默认值不同。

| 参数 | 建议设置 |
| --- | --- |
| 采样器 | FL2VA 和 REF2VA 均使用 `lcm`。 |
| 调度器 | 优先 `simple`；`beta` 作为我的测试中的备选。 |
| 总步数 | `9` 步以上；展示工作流使用 `9` 步。这是 Pika 二次采样的实验结论。 |
| `lowres_steps` | `5`；剩余步数用于高分辨率阶段。总步数为 9 时，高分阶段为 4 步。 |

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
