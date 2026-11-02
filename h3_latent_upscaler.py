"""MiniMax H3 3D latent upscaler, adapted from LBH-123-AI's ComfyUI node.

Copyright (c) 2026 LBH-123-AI

Licensed under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import math
import os
import re
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths


_MODEL_FOLDER = "latent_upscale_models"
if _MODEL_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _MODEL_FOLDER,
        os.path.join(folder_paths.models_dir, _MODEL_FOLDER),
    )

_LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848623809814, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
_LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)
_PRECISIONS = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_MODEL_CACHE = None
_MODEL_CACHE_LOCK = threading.RLock()


def scan_models():
    names = [
        name for name in folder_paths.get_filename_list(_MODEL_FOLDER)
        if os.path.splitext(name)[1].lower() in (".safetensors", ".pth")
    ]
    return names or ["(place H3 upscaler in models/latent_upscale_models)"]


def _detect_arch(state_dict):
    config = {
        "in_channels": 24,
        "in_blocks": 12,
        "out_blocks": 12,
        "channels": 512,
        "temporal_every": 2,
        "temporal_kernel": 5,
    }
    if "conv_in.weight" in state_dict:
        config["in_channels"] = state_dict["conv_in.weight"].shape[1]
        config["channels"] = state_dict["conv_in.weight"].shape[0]

    in_ids = set()
    out_ids = set()
    temporal_keys = []
    for key in state_dict:
        match = re.match(r"in_blocks\.(\d+)\.in_layers\.", key)
        if match:
            in_ids.add(int(match.group(1)))
        match = re.match(r"out_blocks\.(\d+)\.in_layers\.", key)
        if match:
            out_ids.add(int(match.group(1)))
        if key.endswith("dwconv.weight"):
            temporal_keys.append(key)

    if in_ids:
        config["in_blocks"] = len(in_ids)
    if out_ids:
        config["out_blocks"] = len(out_ids)
    if temporal_keys:
        config["temporal_kernel"] = state_dict[temporal_keys[0]].shape[2]
    else:
        config["temporal_every"] = 0
    return config


def _install_cpu_layer_offload(model, execution_device):
    execution_device = torch.device(execution_device)
    lock = threading.RLock()
    handles = []
    refs = []
    max_module_bytes = 0

    for module in model.modules():
        cpu_params = {
            name: param.detach()
            for name, param in module.named_parameters(recurse=False)
        }
        cpu_buffers = {
            name: buffer.detach()
            for name, buffer in module.named_buffers(recurse=False)
            if buffer is not None
        }
        if not cpu_params and not cpu_buffers:
            continue

        module_bytes = sum(tensor.numel() * tensor.element_size()
                           for tensor in (*cpu_params.values(), *cpu_buffers.values()))
        max_module_bytes = max(max_module_bytes, module_bytes)
        refs.append((module, cpu_params, cpu_buffers))

        def page_in(mod, args, parameter_refs=cpu_params, buffer_refs=cpu_buffers):
            lock.acquire()
            try:
                params = {
                    name: tensor.to(device=execution_device)
                    for name, tensor in parameter_refs.items()
                }
                buffers = {
                    name: tensor.to(device=execution_device)
                    for name, tensor in buffer_refs.items()
                }
                for name, tensor in params.items():
                    mod._parameters[name].data = tensor
                for name, tensor in buffers.items():
                    mod._buffers[name] = tensor
            except BaseException:
                lock.release()
                raise

        def page_out(mod, args, output, parameter_refs=cpu_params, buffer_refs=cpu_buffers):
            try:
                for name, param in mod._parameters.items():
                    if param is not None and name in parameter_refs:
                        param.data = parameter_refs[name]
                for name, buffer in mod._buffers.items():
                    if buffer is not None and name in buffer_refs:
                        mod._buffers[name] = buffer_refs[name]
            finally:
                lock.release()
            return output

        handles.append(module.register_forward_pre_hook(page_in))
        handles.append(module.register_forward_hook(page_out, always_call=True))

    model._sol_attn_cpu_layer_handles = handles
    model._sol_attn_cpu_layer_refs = refs
    model._sol_attn_cpu_layer_lock = lock
    return max_module_bytes


def _restore_cpu_layer_offload(model, remove_hooks=False):
    if not hasattr(model, "_sol_attn_cpu_layer_handles"):
        return

    with model._sol_attn_cpu_layer_lock:
        for module, cpu_params, cpu_buffers in model._sol_attn_cpu_layer_refs:
            for name, param in module._parameters.items():
                if param is not None and name in cpu_params:
                    param.data = cpu_params[name]
            for name, buffer in module._buffers.items():
                if buffer is not None and name in cpu_buffers:
                    module._buffers[name] = cpu_buffers[name]

    if remove_hooks:
        for handle in model._sol_attn_cpu_layer_handles:
            handle.remove()
        del model._sol_attn_cpu_layer_handles
        del model._sol_attn_cpu_layer_refs
        del model._sol_attn_cpu_layer_lock


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * channels),
        )
        self.out_norm = nn.GroupNorm(32, channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Identity(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(h)


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.GroupNorm(32, channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        h = F.silu(self.norm(x))
        return x + self.pwconv(self.dwconv(h))


class LatentResizer3D(nn.Module):
    def __init__(self, in_channels, in_blocks, out_blocks, channels, temporal_every, temporal_kernel):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.temporal_kernel = temporal_kernel if temporal_every > 0 else 0
        self.in_blocks = nn.ModuleList()
        for index in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim))
            if temporal_every > 0 and index % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for index in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim))
            if temporal_every > 0 and index % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def temporal_chunk_settings(self):
        chunk = getattr(self, "_sol_attn_temporal_chunk_override", None)
        return max(8, int(chunk if chunk is not None else 32)), self.temporal_kernel

    def temporal_window_budget(self, frames, enable_chunking):
        chunk, overlap = self.temporal_chunk_settings()
        if not enable_chunking or frames <= chunk:
            return frames
        return min(frames + 2 * overlap, chunk + 4 * overlap)

    def forward(self, x, scale, target_size, enable_chunking=True):
        size = target_size
        if size == x.shape[-3:]:
            return x

        frames = x.shape[2]
        chunk_size, overlap = self.temporal_chunk_settings()
        if not enable_chunking or frames <= chunk_size:
            return self._forward_segment(x, scale, size)

        padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        output = x.new_zeros((x.shape[0], x.shape[1], frames, size[-2], size[-1]))
        weights = x.new_zeros((1, 1, frames, 1, 1))

        start = 0
        while start < frames:
            segment_end = min(frames, start + chunk_size)
            output_start = max(0, start - overlap)
            output_end = min(frames, segment_end + overlap)
            low = max(0, output_start - overlap)
            high = min(frames + 2 * overlap, output_end + overlap)
            segment = padded[:, :, low:high].contiguous()
            segment_size = (high - low, size[-2], size[-1])
            segment_output = self._forward_segment(segment, scale, segment_size)

            slice_start = output_start + overlap - low
            valid_output = segment_output[:, :, slice_start:slice_start + output_end - output_start]
            valid_frames = output_end - output_start
            weight = x.new_ones((valid_frames,))
            if start > output_start:
                blend = start - output_start
                weight[:blend] = torch.arange(1, blend + 1, device=x.device, dtype=x.dtype) / (blend + 1)
            if output_end > segment_end:
                blend = output_end - segment_end
                weight[-blend:] = torch.arange(blend, 0, -1, device=x.device, dtype=x.dtype) / (blend + 1)
            weight = weight.view(1, 1, valid_frames, 1, 1)
            output[:, :, output_start:output_end] += valid_output * weight
            weights[:, :, output_start:output_end] += weight
            start += chunk_size

        return output / weights.clamp(min=1e-8)

    def _forward_segment(self, x, scale, size):
        scale_emb = x.new_full((1, 1), scale - 1)
        emb = self.embed(scale_emb)
        x = self.conv_in(x)
        for block in self.in_blocks:
            if isinstance(block, ResBlockEmb3D):
                x = block(x, emb.expand(x.shape[0], -1))
            else:
                x = block(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            if isinstance(block, ResBlockEmb3D):
                x = block(x, emb.expand(x.shape[0], -1))
            else:
                x = block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


def _discard_cached_model(patcher):
    if patcher is None:
        return
    if getattr(patcher, "sol_attn_layerwise_cpu_offload", False):
        _restore_cpu_layer_offload(patcher.model, remove_hooks=True)
    else:
        comfy.model_management.unload_model_and_clones(
            patcher, unload_additional_models=False, all_devices=True
        )


def load_model_patcher(name, precision, layerwise_cpu_offload=False):
    global _MODEL_CACHE
    if not name or name.startswith("("):
        raise FileNotFoundError("Put a MiniMax H3 upscaler checkpoint in models/latent_upscale_models.")

    path = folder_paths.get_full_path_or_raise(_MODEL_FOLDER, name)
    stat = os.stat(path)
    execution_device = comfy.model_management.get_torch_device()
    cache_key = (os.path.normcase(os.path.abspath(path)), stat.st_mtime_ns, stat.st_size,
                 precision, str(execution_device), bool(layerwise_cpu_offload))
    with _MODEL_CACHE_LOCK:
        if _MODEL_CACHE is not None and _MODEL_CACHE[0] == cache_key:
            patcher = _MODEL_CACHE[1]
            logging.info("[Sol-Sampler] Upscaler cache hit: model=%s precision=%s device=%s",
                         name, precision, execution_device)
            return patcher
        if _MODEL_CACHE is not None:
            _discard_cached_model(_MODEL_CACHE[1])
            _MODEL_CACHE = None

        load_started = time.perf_counter()
        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        if any(key.startswith("upscaler.") for key in state_dict):
            state_dict = {key[len("upscaler."):]: value for key, value in state_dict.items() if key.startswith("upscaler.")}
        state_dict = {
            key: value.to(torch.float16) if value.dtype == torch.float8_e4m3fn else value
            for key, value in state_dict.items()
        }

        config = _detect_arch(state_dict)
        with torch.device("meta"):
            model = LatentResizer3D(**config)
        model.load_state_dict(state_dict, strict=True, assign=True)

        if precision == "checkpoint":
            dtype = state_dict["conv_in.weight"].dtype
        else:
            dtype = _PRECISIONS[precision]
            model.to(dtype=dtype)
        model.compute_dtype = dtype
        model.eval()

        max_module_bytes = 0
        use_layerwise_cpu_offload = layerwise_cpu_offload and execution_device.type != "cpu"
        if use_layerwise_cpu_offload:
            max_module_bytes = _install_cpu_layer_offload(model, execution_device)
        patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=torch.device("cpu") if use_layerwise_cpu_offload else execution_device,
            offload_device=torch.device("cpu"),
            size=comfy.model_management.module_size(model),
        )
        patcher.sol_attn_execution_device = execution_device
        patcher.sol_attn_max_module_bytes = max_module_bytes
        patcher.sol_attn_layerwise_cpu_offload = use_layerwise_cpu_offload
        patcher.sol_attn_model_name = name
        _MODEL_CACHE = (cache_key, patcher)
        logging.info(
            "[Sol-Sampler] Upscaler loaded: model=%s precision=%s device=%s mode=%s elapsed=%.3fs",
            name, precision, execution_device,
            "layerwise_cpu" if use_layerwise_cpu_offload else "model_management",
            time.perf_counter() - load_started,
        )
        return patcher


def _inference_memory_required(model, latent, target_size, enable_chunking):
    frames = 1 if latent.ndim == 4 else latent.shape[2]
    temporal_window = model.temporal_window_budget(frames, enable_chunking)
    feature_elements = (
        latent.shape[0] * model.conv_in.out_channels * temporal_window
        * target_size[0] * target_size[1]
    )
    return feature_elements * model.conv_in.weight.element_size() * 8


def _upscaler_memory_targets(patcher, workspace):
    device = patcher.sol_attn_execution_device
    mm = comfy.model_management
    inference_budget = max(
        mm.minimum_inference_memory(), workspace + mm.extra_reserved_memory()
    )
    if patcher.sol_attn_layerwise_cpu_offload:
        upscaler_memory = patcher.sol_attn_max_module_bytes
    else:
        upscaler_memory = mm.LoadedModel(patcher).model_memory_required(device)
    formula_target = math.ceil(upscaler_memory * 1.1 + inference_budget)
    empirical_floor = 0
    total_memory = 0
    if device.type == "cuda":
        _, total_memory = torch.cuda.mem_get_info(device)
        if 10 * 2**30 <= total_memory <= 16 * 2**30:
            empirical_floor = math.ceil(total_memory * 0.607)
    return formula_target, empirical_floor, total_memory, upscaler_memory, inference_budget


def _reserve_upscaler_memory(patcher, workspace, target_free):
    device = patcher.sol_attn_execution_device
    if device.type == "cpu":
        return True, 0, 0, target_free

    mm = comfy.model_management
    formula_target, empirical_floor, total_memory, upscaler_memory, inference_budget = (
        _upscaler_memory_targets(patcher, workspace)
    )
    target_free = max(formula_target, math.ceil(target_free))
    free_before, torch_before = mm.get_free_memory(device, torch_free_too=True)
    deficit = max(0, target_free - free_before)
    if deficit:
        mm.free_memory(target_free, device)
    free_after, torch_after = mm.get_free_memory(device, torch_free_too=True)
    logging.debug(
        "[Sol-Sampler] Upscaler memory: target_free=%.2f MiB formula_target=%.2f MiB "
        "empirical_floor=%.2f MiB upscaler_weights=%.2f MiB workspace_estimate=%.2f MiB "
        "inference_budget=%.2f MiB total_vram=%.2f MiB free_before=%.2f MiB "
        "torch_cache_before=%.2f MiB free_after=%.2f MiB torch_cache_after=%.2f MiB",
        target_free / 2**20, formula_target / 2**20, empirical_floor / 2**20,
        upscaler_memory / 2**20, workspace / 2**20, inference_budget / 2**20,
        total_memory / 2**20, free_before / 2**20, torch_before / 2**20,
        free_after / 2**20, torch_after / 2**20,
    )
    return free_after >= target_free, free_after, empirical_floor, target_free


def upscale_latent(patcher, latent, target_size, scale, enable_chunking):
    upscale_started = time.perf_counter()
    model = patcher.model
    execution_device = patcher.sol_attn_execution_device
    original_chunk_override = getattr(model, "_sol_attn_temporal_chunk_override", None)
    chunk, _ = model.temporal_chunk_settings()
    frames = 1 if latent.ndim == 4 else latent.shape[2]
    overlap = model.temporal_kernel
    chunk_count = math.ceil(frames / chunk) if enable_chunking and frames > chunk else 1
    initial_workspace = _inference_memory_required(model, latent, target_size, enable_chunking)
    logging.info(
        "[Sol-Sampler] H3 upscale started: model=%s precision=%s input=%s target_hw=%s scale=%.3f "
        "mode=%s chunk=%d overlap=%d chunks=%d estimated_workspace=%.2f MiB",
        patcher.sol_attn_model_name, str(model.compute_dtype).split(".")[-1],
        tuple(latent.shape), target_size, scale,
        "chunked" if chunk_count > 1 else "full", chunk,
        overlap if chunk_count > 1 else 0, chunk_count,
        initial_workspace / 2**20,
    )
    formula_target, empirical_floor, total_memory, _, _ = _upscaler_memory_targets(
        patcher, initial_workspace
    )
    target_free = formula_target
    retry_step = max(512 * 2**20, math.ceil(total_memory * 0.08))
    max_attempts = 12
    result = None
    completed_attempt = 0
    completed_chunks = 0
    was_image = latent.ndim == 4
    try:
        for attempt in range(1, max_attempts + 1):
            model._sol_attn_temporal_chunk_override = chunk
            memory_required = _inference_memory_required(
                model, latent, target_size, enable_chunking
            )
            source = mean = std = None
            free_after = 0
            try:
                target_met, free_after, empirical_floor, target_free = _reserve_upscaler_memory(
                    patcher, memory_required, target_free
                )
                attempt_chunk_count = math.ceil(frames / chunk) if enable_chunking and frames > chunk else 1
                logging.debug(
                    "[Sol-Sampler] H3 upscale attempt: model=%s attempt=%d/%d chunk=%d chunks=%d "
                    "max_input_window=%d estimated_workspace=%.2f MiB",
                    patcher.sol_attn_model_name, attempt, max_attempts, chunk, attempt_chunk_count,
                    model.temporal_window_budget(frames, enable_chunking), memory_required / 2**20,
                )
                if not target_met and enable_chunking and chunk > 8:
                    previous_chunk = chunk
                    chunk = max(8, chunk // 2)
                    model._sol_attn_temporal_chunk_override = chunk
                    next_workspace = _inference_memory_required(
                        model, latent, target_size, enable_chunking
                    )
                    next_formula_target, _, _, _, _ = _upscaler_memory_targets(
                        patcher, next_workspace
                    )
                    target_free = max(next_formula_target, free_after)
                    logging.warning(
                        "[Sol-Sampler] H3 free-VRAM target unmet on attempt=%d/%d; "
                        "reduce chunk %d->%d (workspace=%.2f MiB)",
                        attempt, max_attempts, previous_chunk, chunk, next_workspace / 2**20,
                    )
                    continue

                if not patcher.sol_attn_layerwise_cpu_offload:
                    comfy.model_management.load_models_gpu(
                        [patcher], memory_required=memory_required
                    )
                was_image = latent.ndim == 4
                source = latent.unsqueeze(2) if was_image else latent
                source = source.to(device=execution_device, dtype=model.compute_dtype)
                mean = source.new_tensor(_LATENTS_MEAN).view(1, -1, 1, 1, 1)
                std = source.new_tensor(_LATENTS_STD).view(1, -1, 1, 1, 1)
                output = model(
                    (source - mean) / std,
                    scale=scale,
                    target_size=(source.shape[2], target_size[0], target_size[1]),
                    enable_chunking=enable_chunking,
                )
                source = None
                result = output
                completed_attempt = attempt
                completed_chunks = attempt_chunk_count
                break
            except torch.OutOfMemoryError as exc:
                source = mean = std = None
                release_upscaler_model(patcher)
                if attempt >= max_attempts:
                    logging.error(
                        "[Sol-Sampler] H3 upscale failed after %d attempts at chunk=%d: %s",
                        attempt, chunk, str(exc).splitlines()[0],
                    )
                    raise
                if target_free < empirical_floor:
                    target_free = min(empirical_floor, target_free + retry_step)
                    action = f"increase free-VRAM target to {target_free / 2**20:.0f} MiB"
                elif enable_chunking and chunk > 8:
                    previous_chunk = chunk
                    chunk = max(8, chunk // 2)
                    model._sol_attn_temporal_chunk_override = chunk
                    next_workspace = _inference_memory_required(
                        model, latent, target_size, enable_chunking
                    )
                    next_formula_target, _, _, _, _ = _upscaler_memory_targets(
                        patcher, next_workspace
                    )
                    target_free = max(next_formula_target, free_after)
                    action = f"shrink temporal chunk {previous_chunk}->{chunk}"
                else:
                    logging.error(
                        "[Sol-Sampler] H3 upscale failed at chunk=%d: %s",
                        chunk, str(exc).splitlines()[0],
                    )
                    raise
                logging.warning(
                    "[Sol-Sampler] H3 OOM on attempt=%d/%d (%s); %s",
                    attempt, max_attempts, str(exc).splitlines()[0], action,
                )
    finally:
        model._sol_attn_temporal_chunk_override = original_chunk_override
        _restore_cpu_layer_offload(model)
    result.mul_(std).add_(mean)
    if was_image:
        result = result.squeeze(2)
    result = result.to(device=latent.device, dtype=latent.dtype)
    logging.info(
        "[Sol-Sampler] H3 upscale completed: attempts=%d/%d chunk=%d chunks=%d elapsed=%.3fs",
        completed_attempt, max_attempts, chunk, completed_chunks,
        time.perf_counter() - upscale_started,
    )
    return result


def release_upscaler_model(patcher):
    if patcher is None:
        return
    if patcher.sol_attn_layerwise_cpu_offload:
        _restore_cpu_layer_offload(patcher.model)
        return
    comfy.model_management.unload_model_and_clones(
        patcher, unload_additional_models=False, all_devices=True
    )
