# Copyright 2026 ANe5s
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import time

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.utils
from tqdm.auto import tqdm
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .h3_latent_upscaler import load_model_patcher, release_upscaler_model, scan_models, upscale_latent


class _StageTimer:
    def __init__(self, stage):
        self.stage = stage
        # Avoid a CUDA sync solely for logging.
        self.started = time.perf_counter()
        self.elapsed = None

    def finish(self, status="completed"):
        if self.elapsed is None:
            self.elapsed = time.perf_counter() - self.started
            logging.info("[Sol-Sampler] %s %s in %.3fs", self.stage, status, self.elapsed)
        return self.elapsed


class _TqdmLogStream:
    encoding = "utf-8"

    def write(self, text):
        for line in text.split("\r"):
            line = line.strip()
            if line:
                logging.info("%s", line)
        return len(text)

    def flush(self):
        pass


def _unpack(x, shapes):
    if len(shapes) > 1:
        return comfy.utils.unpack_latents(x, shapes)
    return [x]


def _pack(parts, shapes):
    if len(shapes) > 1:
        return comfy.utils.pack_latents(parts)[0]
    return parts[0]


def _spatial_streams(shapes):
    if len(shapes) == 1 and len(shapes[0]) in (4, 5):
        return {0}
    return {i for i, shape in enumerate(shapes) if len(shape) == 5}


def _resize_spatial(x, size, mode):
    if x.ndim == 4:
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        return F.interpolate(x.float(), size=size, mode=mode, **kwargs).to(x.dtype)
    if x.ndim == 5:
        batch, channels, frames, height, width = x.shape
        video = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        video = F.interpolate(video.float(), size=size, mode=mode, **kwargs).to(x.dtype)
        return video.reshape(batch, frames, channels, *size).permute(0, 2, 1, 3, 4)
    raise ValueError(f"Sol-Attn Progressive Sampler cannot resize a {x.ndim}D latent stream")


def _scaled_shapes(shapes, spatial_streams, scale):
    result = []
    for i, shape in enumerate(shapes):
        shape = list(shape)
        if i in spatial_streams:
            shape[-2] = max(1, round(shape[-2] * scale))
            shape[-1] = max(1, round(shape[-1] * scale))
        result.append(tuple(shape))
    return result


def _resize_latent(x, source_shapes, target_shapes, spatial_streams, mode):
    parts = _unpack(x, source_shapes)
    resized = []
    for i, part in enumerate(parts):
        if i in spatial_streams:
            resized.append(_resize_spatial(part, target_shapes[i][-2:], mode))
        else:
            resized.append(part)
    return _pack(resized, target_shapes)


def _upscale_denoised(x, source_shapes, target_shapes, spatial_streams, mode, interpolation,
                      upscaler_patcher, temporal_chunking):
    if mode != "Minimax H3 3D" or upscaler_patcher is None:
        return _resize_latent(x, source_shapes, target_shapes, spatial_streams, interpolation)

    parts = _unpack(x, source_shapes)
    upscaled = []
    for i, part in enumerate(parts):
        if i not in spatial_streams:
            upscaled.append(part)
            continue
        source_h, source_w = source_shapes[i][-2:]
        target_h, target_w = target_shapes[i][-2:]
        scale = ((target_h / source_h) + (target_w / source_w)) / 2.0
        upscaled.append(
            upscale_latent(
                upscaler_patcher,
                part,
                (target_h, target_w),
                scale,
                temporal_chunking,
            )
        )
    return _pack(upscaled, target_shapes)


def _stage_conds(guider, base_model, noise, latent, mask, shapes, seed):
    conds = {
        key: [cond.copy() for cond in values]
        for key, values in guider.original_conds.items()
    }
    comfy.samplers.preprocess_conds_hooks(conds)
    conds = comfy.samplers.process_conds(
        base_model, noise, conds, noise.device, latent, mask, seed, latent_shapes=shapes
    )
    comfy.samplers.filter_registered_hooks_on_conds(conds, guider.model_options)
    return conds


def _notify(callback, i, sigma, denoised, x):
    if callback is not None:
        callback({
            "i": i,
            "denoised": denoised,
            "x": x,
            "sigma": sigma,
            "sigma_hat": sigma,
        })


def _sample_with_native(sampler, model, x, sigmas, extra_args, callback, disable, step_offset=0,
                        progress_total=None):
    progress = None
    sampler_disable = disable
    if progress_total is not None:
        progress = tqdm(
            total=progress_total,
            disable=bool(disable),
            mininterval=0,
            miniters=1,
            file=_TqdmLogStream(),
            leave=False,
            ascii=False,
        )
        sampler_disable = True

    def sampler_callback(data):
        i = data["i"]
        sigma = data["sigma"] if "sigma" in data else sigmas[i]
        if callback is not None:
            callback({
                "i": i + step_offset,
                "sigma": sigma,
                "sigma_hat": data.get("sigma_hat", sigma),
                "denoised": data["denoised"],
                "x": data["x"],
            })
        if progress is not None:
            progress.update()

    try:
        return sampler.sampler_function(
            model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=sampler_callback if callback is not None or progress is not None else None,
            disable=sampler_disable,
            **sampler.extra_options,
        )
    finally:
        if progress is not None:
            progress.close()


def _native_inpaint_noise(sampler, noise, extra_args):
    if not sampler.inpaint_options.get("random", False):
        return noise
    generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
    return torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)


def _progressive_sampler(sampler_name, sampler, lowres_steps, lowres_scale, upsample,
                         upscaler_model, upscaler_precision, temporal_chunking,
                         layerwise_cpu_offload):
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        sampling_timer = _StageTimer("Sampling")
        sampling_completed = False
        extra_args = {} if extra_args is None else extra_args
        interpolation = upsample if upsample in ("bilinear", "nearest") else "bilinear"
        use_h3_upscaler = upsample == "Minimax H3 3D" and upscaler_model not in (None, "none")
        upscale_label = upsample if upsample != "Minimax H3 3D" or use_h3_upscaler else "bilinear (upscaler=none)"
        high_noise = model.noise
        high_inpaint_noise = _native_inpaint_noise(sampler, high_noise, extra_args)
        total_steps = sigmas.numel() - 1
        progress_supported = sampler_name not in ("dpm_fast", "dpm_adaptive")
        full_progress_total = total_steps if progress_supported else None
        if lowres_scale == 1.0:
            logging.info("[Sol-Sampler] Sampling started: sampler=%s mode=single-pass steps=%d",
                         sampler_name, total_steps)
            model.noise = high_inpaint_noise
            try:
                result = _sample_with_native(
                    sampler, model, x, sigmas, extra_args, callback, disable,
                    progress_total=full_progress_total,
                )
                sampling_completed = True
                return result
            finally:
                model.noise = high_noise
                sampling_timer.finish("completed" if sampling_completed else "stopped")

        if total_steps < 2:
            logging.info(
                "[Sol-Sampler] Sampling started: sampler=%s mode=single-pass steps=%d reason=not-enough-steps",
                sampler_name, total_steps,
            )
            model.noise = high_inpaint_noise
            try:
                result = _sample_with_native(
                    sampler, model, x, sigmas, extra_args, callback, disable,
                    progress_total=full_progress_total,
                )
                sampling_completed = True
                return result
            finally:
                model.noise = high_noise
                sampling_timer.finish("completed" if sampling_completed else "stopped")

        low_steps = min(lowres_steps, total_steps - 1)
        guider = model.inner_model
        base_model = guider.inner_model
        high_shapes = list(base_model.latent_shapes or [tuple(x.shape)])
        spatial_streams = _spatial_streams(high_shapes)
        if not spatial_streams:
            raise ValueError("Sol-Attn Progressive Sampler needs an image or video latent stream")

        low_shapes = _scaled_shapes(high_shapes, spatial_streams, lowres_scale)
        if low_shapes == high_shapes:
            logging.info(
                "[Sol-Sampler] Sampling started: sampler=%s mode=single-pass steps=%d reason=latent-size-unchanged",
                sampler_name, total_steps,
            )
            model.noise = high_inpaint_noise
            try:
                result = _sample_with_native(
                    sampler, model, x, sigmas, extra_args, callback, disable,
                    progress_total=full_progress_total,
                )
                sampling_completed = True
                return result
            finally:
                model.noise = high_noise
                sampling_timer.finish("completed" if sampling_completed else "stopped")

        logging.info(
            "[Sol-Sampler] Sampling started: sampler=%s upscale=%s lowres=%d/%d steps "
            "scale=%.2fx highres=%d steps",
            sampler_name, upscale_label, low_steps, total_steps, lowres_scale, total_steps - low_steps,
        )
        high_latent = model.latent_image
        high_mask = extra_args.get("denoise_mask")
        high_conds = guider.conds
        nested = len(high_shapes) > 1

        low_prepare_timer = _StageTimer("Low-resolution preparation")
        low_noise = _resize_latent(high_noise, high_shapes, low_shapes, spatial_streams, interpolation)
        low_inpaint_noise = _resize_latent(
            high_inpaint_noise, high_shapes, low_shapes, spatial_streams, interpolation
        )
        low_latent = _resize_latent(high_latent, high_shapes, low_shapes, spatial_streams, interpolation)
        if high_mask is not None:
            if nested:
                low_mask = _resize_latent(high_mask, high_shapes, low_shapes, spatial_streams, "bilinear")
            else:
                low_mask = _resize_spatial(high_mask, low_shapes[0][-2:], "bilinear")
        else:
            low_mask = None
        low_x = _resize_latent(x, high_shapes, low_shapes, spatial_streams, interpolation)

        try:
            base_model.latent_shapes = low_shapes
            model.noise = low_inpaint_noise
            model.latent_image = low_latent
            if "denoise_mask" in extra_args:
                extra_args["denoise_mask"] = low_mask
            guider.conds = _stage_conds(
                guider, base_model, low_noise, low_latent, low_mask, low_shapes, extra_args.get("seed")
            )
            low_prepare_timer.finish()

            denoised_low = None
            transition_sigma = None
            transition_done = False
            low_timer = _StageTimer("Low-resolution denoising")
            logging.info("[Sol-Sampler] Low-resolution sampling started (%d steps)", low_steps)

            def low_callback(data):
                nonlocal denoised_low, transition_sigma, transition_done
                i = data["i"]
                sigma = data["sigma"]
                at_low_boundary = i == low_steps - 1
                if sampler_name == "dpm_adaptive":
                    at_low_boundary = math.isclose(
                        float(sigma), float(sigmas[low_steps]), rel_tol=1e-5, abs_tol=1e-8
                    )

                if at_low_boundary and not transition_done:
                    transition_done = True
                    denoised_low = data["denoised"]
                    transition_sigma = sigma
                else:
                    _notify(
                        callback,
                        i,
                        sigma,
                        _resize_latent(data["denoised"], low_shapes, high_shapes, spatial_streams, interpolation),
                        _resize_latent(data["x"], low_shapes, high_shapes, spatial_streams, interpolation),
                    )

            low_sampling_completed = False
            try:
                low_x = _sample_with_native(
                    sampler,
                    model,
                    low_x,
                    sigmas[:low_steps + 1],
                    extra_args,
                    low_callback,
                    disable,
                    progress_total=low_steps if progress_supported else None,
                )
                low_sampling_completed = True
            finally:
                low_timer.finish("completed" if low_sampling_completed else "stopped")

            transition_timer = _StageTimer("Latent transition")
            upscaler_patcher = None
            transition_completed = False
            try:
                try:
                    if use_h3_upscaler:
                        upscaler_patcher = load_model_patcher(
                            upscaler_model, upscaler_precision, layerwise_cpu_offload
                        )
                    denoised_high = _upscale_denoised(
                        denoised_low,
                        low_shapes,
                        high_shapes,
                        spatial_streams,
                        upsample,
                        interpolation,
                        upscaler_patcher,
                        temporal_chunking,
                    )
                    denoised_low = None
                finally:
                    release_upscaler_model(upscaler_patcher)
                transition_completed = True
            finally:
                transition_timer.finish("completed" if transition_completed else "stopped")
            if high_mask is not None:
                denoised_high = denoised_high * high_mask + high_latent * (1.0 - high_mask)
            preview_x = _resize_latent(low_x, low_shapes, high_shapes, spatial_streams, interpolation)
            _notify(callback, low_steps - 1, transition_sigma, denoised_high, preview_x)

            high_setup_timer = _StageTimer("High-resolution setup")
            low_parts = _unpack(low_x, low_shapes)
            noise_parts = _unpack(high_noise, high_shapes)
            clean_parts = _unpack(denoised_high, high_shapes)
            sigma_next = sigmas[low_steps]
            next_parts = []
            for j, state in enumerate(low_parts):
                if j in spatial_streams:
                    state = base_model.model_sampling.noise_scaling(
                        sigma_next, noise_parts[j], clean_parts[j]
                    )
                next_parts.append(state)
            x = _pack(next_parts, high_shapes)

            base_model.latent_shapes = high_shapes
            model.noise = high_inpaint_noise
            model.latent_image = high_latent
            if "denoise_mask" in extra_args:
                extra_args["denoise_mask"] = high_mask
            guider.conds = high_conds

            guider.inner_model, guider.conds, high_loaded_models = comfy.sampler_helpers.prepare_sampling(
                guider.model_patcher, x.shape, guider.conds, guider.model_options
            )
            loaded_models = {id(patcher): patcher for patcher in guider.loaded_models}
            loaded_models.update((id(patcher), patcher) for patcher in high_loaded_models)
            guider.loaded_models = list(loaded_models.values())
            base_model = guider.inner_model
            base_model.latent_shapes = high_shapes
            high_setup_timer.finish()

            highres_steps = total_steps - low_steps
            high_timer = _StageTimer("High-resolution denoising")
            logging.info("[Sol-Sampler] High-resolution sampling started (%d steps)", highres_steps)
            try:
                result = _sample_with_native(
                    sampler, model, x, sigmas[low_steps:], extra_args,
                    callback, disable, low_steps,
                    progress_total=highres_steps if progress_supported else None,
                )
                sampling_completed = True
                return result
            finally:
                high_timer.finish("completed" if sampling_completed else "stopped")
        finally:
            base_model.latent_shapes = high_shapes
            model.noise = high_noise
            model.latent_image = high_latent
            if "denoise_mask" in extra_args:
                extra_args["denoise_mask"] = high_mask
            guider.conds = high_conds
            sampling_timer.finish("completed" if sampling_completed else "stopped")

    return sample


class SolAttnProgressiveSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        upscaler_models = ["none"] + [name for name in scan_models() if not name.startswith("(")]
        default_upscaler = upscaler_models[1] if len(upscaler_models) > 1 else "none"
        return io.Schema(
            node_id="SolAttnProgressiveSampler",
            display_name="Sol-Attn Progressive Sampler",
            category="ANe5s Nodes/Sol-Sampler",
            description=(
                "Runs low-resolution sampling, spatial latent upscaling, and high-resolution sampling "
                "inside one native sampler call. Connect a model patched by Model Sparse Attention (sol-attn) "
                "to the guider, then use this SAMPLER with SamplerCustomAdvanced."
            ),
            inputs=[
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="euler",
                               tooltip="Native ComfyUI sampler used for both resolution stages."),
                io.Int.Input("lowres_steps", default=6, min=1, max=10000,
                             tooltip="Number of denoiser evaluations at low resolution. At least one step is reserved for the high-resolution stage."),
                io.Float.Input("lowres_scale", default=0.5, min=0.25, max=1.0, step=0.05,
                               tooltip="Set to 1.0 to run one native sampling pass without latent upscaling or a second stage."),
                io.Combo.Input("latent_upsample", options=["bilinear", "nearest", "Minimax H3 3D"], default="Minimax H3 3D",
                               tooltip="How to upscale the low-resolution denoised latent. If Minimax H3 3D is selected with upscaler_model=none, bilinear interpolation is used. Noise, masks, and previews use bilinear/nearest interpolation."),
                io.Combo.Input("upscaler_model", options=upscaler_models, default=default_upscaler, optional=True,
                               tooltip="MiniMax H3 3D checkpoint in models/latent_upscale_models. Select none to use bilinear interpolation instead of the H3 model."),
                io.Combo.Input("upscaler_precision", options=["checkpoint", "fp32", "fp16", "bf16"],
                               default="checkpoint", optional=True,
                               tooltip="Inference dtype for the learned upscaler."),
                io.Boolean.Input("upscaler_temporal_chunking", default=True, optional=True,
                                 tooltip="Split video latents into overlapping temporal chunks when they exceed the H3 upscaler's chunk size, then blend overlaps to reduce peak VRAM. Audio latents are not chunked."),
                io.Boolean.Input("upscaler_layerwise_cpu_offload", default=False, optional=True,
                                 tooltip="Keep H3 upscaler weights on CPU and move each module to the GPU only for its forward pass. This reduces peak VRAM but adds CPU/GPU transfer overhead."),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, sampler_name, lowres_steps, lowres_scale, latent_upsample, upscaler_model=None,
                upscaler_precision="checkpoint", upscaler_temporal_chunking=True,
                upscaler_layerwise_cpu_offload=False) -> io.NodeOutput:
        native_sampler = comfy.samplers.sampler_object(sampler_name)
        sampler = comfy.samplers.KSAMPLER(
            _progressive_sampler(
                sampler_name,
                native_sampler,
                lowres_steps,
                lowres_scale,
                latent_upsample,
                upscaler_model,
                upscaler_precision,
                upscaler_temporal_chunking,
                upscaler_layerwise_cpu_offload,
            )
        )
        return io.NodeOutput(sampler)


class SolSamplerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SolAttnProgressiveSampler]


async def comfy_entrypoint() -> SolSamplerExtension:
    return SolSamplerExtension()
