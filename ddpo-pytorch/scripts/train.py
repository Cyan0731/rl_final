from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
import numpy as np
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from transformers import ClapTextModelWithProjection
import diffusers
from diffusers import AudioLDMPipeline
from pipeline_audioldm2 import AudioLDM2Pipeline
from transformers import ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan

# from diffusers.optimization import get_scheduler
# from diffusers.utils import check_min_version, is_wandb_available
# from diffusers.utils.import_utils import is_xformers_available
# from audioldm.audio import TacotronSTFT, read_wav_file
# from audioldm.utils import default_audioldm_config
# import matplotlib
# matplotlib.use('Agg') # No pictures displayed 
# import matplotlib.pyplot as plt
# import pylab
# import librosa
# import librosa.display
from ddpo_pytorch.prompts import nouns_activities
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
        * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    # set_seed(config.seed, device_specific=True)
    pretrained_model_name_or_path = "/home/cyan/rl_final/ddpo-pytorch/audioldm2-music"
    # #change stable diffusion to audioldm
    # tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    # noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    # text_encoder = ClapTextModelWithProjection.from_pretrained(
    #     pretrained_model_name_or_path, subfolder="text_encoder", revision=False
    # )
    # vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=False)
    # unet = UNet2DConditionModel.from_pretrained(
    #     pretrained_model_name_or_path, subfolder="unet", revision=False
    # )
    # vocoder = SpeechT5HifiGan.from_pretrained(pretrained_model_name_or_path, subfolder="vocoder", revision=False
    # )

    # audioldmpipeline=AudioLDMPipeline(
    #     text_encoder=text_encoder,
    #     vae=vae,
    #     unet=unet,
    #     vocoder=vocoder,
    #     scheduler=noise_scheduler,
    #     tokenizer=tokenizer
    # ).to(accelerator.device)
    # test if weight is successfully loaded
    # waveform = audioldmpipeline("a High quality music with delighted guitar",negative_prompt="low quality", num_inference_steps=50, num_waveforms_per_prompt=1, audio_length_in_s=20).audios
    # print(waveform)
    # print(waveform.shape)
    # # Convert the numpy ndarray to a PyTorch tensor
    # waveform_tensor = torch.tensor(waveform)
    # print(type(waveform_tensor))
    # # Save the tensor as a WAV file
    # import torchaudio
    # torchaudio.save(f"a High quality music with delighted guitar.wav", waveform_tensor, 16000)
    
    #Try audioldm2 
    audioldmpipeline= AudioLDM2Pipeline.from_pretrained(
        pretrained_model_name_or_path,
    ).to(accelerator.device)
    audioldmpipeline = audioldmpipeline.to("cuda")
    audioldmpipeline.vae.requires_grad_(False)
    audioldmpipeline.text_encoder.requires_grad_(False)
    audioldmpipeline.text_encoder_2.requires_grad_(False)
    audioldmpipeline.language_model.requires_grad_(False)
    audioldmpipeline.vocoder.requires_grad_(False)
    audioldmpipeline.unet.requires_grad_(not config.use_lora)
    print(audioldmpipeline.unet.config)
    audioldmpipeline.text_encoder.text_projection.requires_grad_(not config.use_lora)
    # # disable safety checker
    # audioldmpipeline.safety_checker = None
    # # make the progress bar nicer
    # audioldmpipeline.set_progress_bar_config(
    #     position=1,
    #     disable=not accelerator.is_local_main_process,
    #     leave=False,
    #     desc="Timestep",
    #     dynamic_ncols=True,
    # )
    # switch to DDIM scheduler
    # audioldmpipeline.scheduler = DDIMScheduler.from_config(audioldmpipeline.scheduler.config)
   

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    audioldmpipeline.vae.to(accelerator.device, dtype=inference_dtype)
    audioldmpipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    audioldmpipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    audioldmpipeline.projection_model.to(accelerator.device, dtype=inference_dtype)
    audioldmpipeline.language_model.to(accelerator.device, dtype=inference_dtype)
    audioldmpipeline.text_encoder.text_projection.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        audioldmpipeline.unet.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        i = 0
        cross = [None,768,1024]
        for name in audioldmpipeline.unet.attn_processors.keys():
            # print(name)
            if name.startswith("mid_block"):
                hidden_size = audioldmpipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(audioldmpipeline.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = audioldmpipeline.unet.config.block_out_channels[block_id]
            # print("hidden_size",hidden_size)
            if name.endswith("attn1.processor"):
                cross_attention_dim = None
            else:
                # if "down_blocks" in name or "up_blocks" in name:
                cross_attention_dim = cross[i%3]
                i = i + 1
            # print(hidden_size)
            # print("cross_attention_dim",cross_attention_dim)
            lora_attn_procs[name] = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank = 8
            )
        audioldmpipeline.unet.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return audioldmpipeline.unet(*args, **kwargs)

        unet = _Wrapper(audioldmpipeline.unet.attn_processors)
    else:
        unet = audioldmpipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            audioldmpipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model,
                revision=config.pretrained.revision,
                subfolder="unet",
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()

    # generate negative prompt embeddings
    # neg_prompt_embed = audioldmpipeline.text_encoder(
    #     audioldmpipeline.tokenizer(
    #         text = ["low quality"],
    #         return_tensors="pt",
    #         padding="max_length",
    #         truncation=True,
    #         max_length=audioldmpipeline.tokenizer.model_max_length,
    #     ).input_ids.to(accelerator.device)
    # )[0]
    # sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    # train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.per_prompt_stat_tracking.buffer_size,
            config.per_prompt_stat_tracking.min_count,
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = (
        config.sample.batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0

    # TODO:Change prompt function from config base.py
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        audioldmpipeline.unet.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            # prompts, prompt_metadata = zip(
            #     *[
            #         prompt_fn(**config.prompt_fn_kwargs)
            #         for _ in range(config.sample.batch_size)
            #     ]
            # )
            prompt, prompt_metadata = zip(
                *[
                    nouns_activities("/home/cyan/rl_final/ddpo-pytorch/ddpo_pytorch/assets/tiny_emos.txt",
                                     "/home/cyan/rl_final/ddpo-pytorch/ddpo_pytorch/assets/tiny_inst.txt")
                    for _ in range(config.sample.batch_size)
                ]
            )
            
            # print(prompt[0])
            # print("prompts",prompts)
            # print("prompt_metadata",prompt_metadata)
            # encode prompts
            # prompt_ids = audioldmpipeline.tokenizer(
            #     prompts,
            #     return_tensors="pt",
            #     padding="max_length",
            #     truncation=True,
            #     max_length=audioldmpipeline.tokenizer.model_max_length,
            # ).input_ids.to(accelerator.device)
            # prompt_embeds = audioldmpipeline.text_encoder(prompt_ids)[0]
            # sample_neg_prompt_embeds = sample_neg_prompt_embeds.squeeze(1)
            # train_neg_prompt_embeds = train_neg_prompt_embeds.squeeze(1)
            # sample
            with autocast():
                audio, latents, log_probs, prompt_embeds, train_neg_prompt_embeds, attention_mask, generated_prompt_embeds = pipeline_with_logprob(
                    audioldmpipeline,
                    prompt=prompt[0],
                    negative_prompt="Low quality, mutiple sound source",
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                )
            # print("prompt",prompt)
            # print("prompt[0]",prompt[0])
            prompt_embeds, train_neg_prompt_embeds = prompt_embeds.chunk(2)
            # prompt_embeds_size = prompt_embeds.size()
            # padding = (0, 0, 0, 18-prompt_embeds_size[1])
            # print(prompt_embeds_size[1])
            # prompt_embeds = torch.nn.functional.pad(prompt_embeds, padding, "constant", 0)
            # train_neg_prompt_embeds = torch.nn.functional.pad(train_neg_prompt_embeds, padding, "constant", 0)
            # print("prompt_embeds",prompt_embeds.size())
            # print("train_neg_prompt_embeds",train_neg_prompt_embeds.size())
            # print("generated_prompt_embeds",generated_prompt_embeds.size())
            # print("attention_mask",attention_mask.size())
            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = audioldmpipeline.scheduler.timesteps.repeat(
                config.sample.batch_size, 1
            )  # (batch_size, num_steps)
            # TODO : change the way of computing reward
            # compute rewards asynchronously
            # print("latents",latents.size())
            # print("audio",audio.size())
            # print("prompt[0] type",type(prompt[0]))
            prompts.append(prompt[0])
            # print("audio",audio.shape)
            # print("prompts",prompts)
            rewards = executor.submit(reward_fn, audio, prompt, prompt_metadata)
            
            # yield to to make sure reward computation starts
            time.sleep(0)
            
            samples.append(
                {
                    # "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )
    
        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            # print("reward",rewards)
            # print("reward.shape",rewards.shape)
            # print("reward type",type(rewards))
            # max_reward = np.max(rewards)
            # print("max_reward", max_reward)
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)
        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        # print(samples)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()
        # print("gathered reward", rewards)
        # log rewards and images
        accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            # prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            # prompts = audioldmpipeline.tokenizer.batch_decode(
            #     prompt_ids, skip_special_tokens=True
            # )
            advantages = stat_tracker.update(prompts, rewards)
            # print("prompts",prompts)
            # print("rewards",rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        # print("advantages", advantages.shape)
        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        del samples["rewards"]
        # del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == config.sample.batch_size * config.sample.num_batches_per_epoch
        )
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            audioldmpipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        with autocast():
                            if config.train.cfg:
                                # print("attention_mask",attention_mask.size())
                                # print("embeds",embeds.size())
                                # print("generated_prompt_embeds",generated_prompt_embeds.size())
                                noise_pred = unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    encoder_hidden_states=generated_prompt_embeds,
                                    encoder_hidden_states_1=embeds,
                                    encoder_attention_mask_1=attention_mask,
                                    return_dict=False,
                                )[0]
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = (
                                    noise_pred_uncond
                                    + config.sample.guidance_scale
                                    * (noise_pred_text - noise_pred_uncond)
                                )
                            else:
                                noise_pred = unet(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    class_labels = embeds,
                                    encoder_hidden_states=None
                                ).sample
                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(
                                audioldmpipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss, retain_graph=True)

                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                unet.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()


if __name__ == "__main__":
    app.run(main)
