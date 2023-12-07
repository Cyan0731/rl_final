# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# with the following modifications:
# - It uses the patched version of `ddim_step_with_logprob` from `ddim_with_logprob.py`. As such, it only supports the
#   `ddim` scheduler.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import numpy as np
# from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
#     StableDiffusionPipeline,
#     rescale_noise_cfg,
# )
from .ddim_with_logprob import ddim_step_with_logprob
from diffusers import AudioLDMPipeline


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

@torch.no_grad()
def pipeline_with_logprob(
    self: AudioLDMPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    audio_length_in_s = 5,
    vocoder_upsample_factor = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_waveforms_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    # 0. Default height and width to unet
    # height = height or self.unet.config.sample_size * self.vae_scale_factor
    vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate
    height = int(audio_length_in_s / vocoder_upsample_factor)
    original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
    if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            print( f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process.")
    # width = width or self.unet.config.sample_size * self.vae_scale_factor


    # print(self.unet.config)
    # print("self.vae_scale_factor",self.vae_scale_factor)
    # print("self.vocoder.config.model_in_dim",self.vocoder.config.model_in_dim)
    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        audio_length_in_s,
        vocoder_upsample_factor,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    # print("batch_size",batch_size)
    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )

    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance ,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    # print("height",height)

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_waveforms_per_prompt,
        num_channels_latents,
        height,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    # print("latents size",latents.size())
    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=None,
                class_labels=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )
            
            # print("latents size",latents.size())
            # compute the previous noisy sample x_t -> x_t-1
            latents, log_prob = ddim_step_with_logprob(
                self.scheduler, noise_pred, t, latents, **extra_step_kwargs
            )
            # print("log_prob",log_prob)
            # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            # print("latents size",latents.size())
            all_latents.append(latents)
            # log_prob = torch.zeros(latents.shape)
            all_log_probs.append(log_prob)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    # if not output_type == "latent":
    mel = self.decode_latents(latents)
    # mel, has_nsfw_concept = self.run_safety_checker(
    #     mel, device, prompt_embeds.dtype
    # )
   
    # mel = mel.squeeze(1)
    # print(mel.size())
    mel = mel.to(dtype=torch.float32, device='cuda') 
    # print(self.vocoder.config)
    audio = self.mel_spectrogram_to_waveform(mel)
    audio = audio[:, :original_waveform_length]
    # else:
    #     mel = latents
    #     has_nsfw_concept = None

    # not sure how to use the below code

    # if has_nsfw_concept is None:
    #     do_denormalize = [True] * image.shape[0]
    # else:
    #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    # image = self.image_processor.postprocess(
    #     image, output_type=output_type, do_denormalize=do_denormalize
    # )

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    return audio, all_latents, all_log_probs, prompt_embeds
    

    #check original

    # vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

    # if audio_length_in_s is None:
    #     audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor

    # height = int(audio_length_in_s / vocoder_upsample_factor)

    # original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
    # if height % self.vae_scale_factor != 0:
    #     height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
    #     print(
    #         f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
    #         f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
    #         f"denoising process."
    #     )

    # # 1. Check inputs. Raise error if not correct
    # self.check_inputs(
    #     prompt,
    #     audio_length_in_s,
    #     vocoder_upsample_factor,
    #     callback_steps,
    #     negative_prompt,
    #     prompt_embeds,
    #     negative_prompt_embeds,
    # )

    # # 2. Define call parameters
    # if prompt is not None and isinstance(prompt, str):
    #     batch_size = 1
    # elif prompt is not None and isinstance(prompt, list):
    #     batch_size = len(prompt)
    # else:
    #     batch_size = prompt_embeds.shape[0]

    # device = self._execution_device
    # # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # # corresponds to doing no classifier free guidance.
    # do_classifier_free_guidance = guidance_scale > 1.0

    # # 3. Encode input prompt
    # prompt_embeds = self._encode_prompt(
    #     prompt,
    #     device,
    #     num_waveforms_per_prompt,
    #     do_classifier_free_guidance,
    #     negative_prompt,
    #     prompt_embeds=prompt_embeds,
    #     negative_prompt_embeds=negative_prompt_embeds,
    # )

    # # 4. Prepare timesteps
    # self.scheduler.set_timesteps(num_inference_steps, device=device)
    # timesteps = self.scheduler.timesteps

    # # 5. Prepare latent variables
    # num_channels_latents = self.unet.config.in_channels
    # latents = self.prepare_latents(
    #     batch_size * num_waveforms_per_prompt,
    #     num_channels_latents,
    #     height,
    #     prompt_embeds.dtype,
    #     device,
    #     generator,
    #     latents,
    # )

    # # 6. Prepare extra step kwargs
    # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # # 7. Denoising loop
    # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    # with self.progress_bar(total=num_inference_steps) as progress_bar:
    #     for i, t in enumerate(timesteps):
    #         # expand the latents if we are doing classifier free guidance
    #         latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    #         latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

    #         # predict the noise residual
    #         noise_pred = self.unet(
    #             latent_model_input,
    #             t,
    #             encoder_hidden_states=None,
    #             class_labels=prompt_embeds,
    #             cross_attention_kwargs=cross_attention_kwargs,
    #         ).sample

    #         # perform guidance
    #         if do_classifier_free_guidance:
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    #         # compute the previous noisy sample x_t -> x_t-1
    #         latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    #         # call the callback, if provided
    #         if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
    #             progress_bar.update()
    #             if callback is not None and i % callback_steps == 0:
    #                 step_idx = i // getattr(self.scheduler, "order", 1)
    #                 callback(step_idx, t, latents)

    # # 8. Post-processing
    # mel_spectrogram = self.decode_latents(latents)
    # mel_spectrogram = mel_spectrogram.to(dtype=torch.float32, device='cuda') 
    # audio = self.mel_spectrogram_to_waveform(mel_spectrogram)

    # audio = audio[:, :original_waveform_length]
    # audio = torch.tensor(audio)
    # print(type(audio))
    # # Save the tensor as a WAV file
    # import torchaudio
    # torchaudio.save(f"a High quality music with delighted guitar.wav", audio, 16000)
    # if output_type == "np":
    #     audio = audio.numpy()

    # if not return_dict:
    #     return (audio,)
