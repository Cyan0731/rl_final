# rl_final


## Preparation

after git clone this repo, cd to `ddpo_pytorch` directory and `pip install -e .` to install the repo.

download CLAP weight below and move CLAP weight into `ddpo-pytorch/CLAP/ckpt` directory
https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt
<br/><br/>

## Training

### Training Settings for DDPO PyTorch Model

This document outlines the necessary settings and steps to train the DDPO PyTorch model. 

<br/><br/>

### Configuration Settings

Several parameters in `config/base.py` can be adjusted for training:

#### Use LoRA
LoRA (Low-Rank Adaptation) can save a significant amount of memory during training.

```python
config.use_lora = True
``` 

#### change denoising step
The num_steps parameter is used for the denoising steps. It is essential to record the log probability (policy) and next latent (action) in the denoising process.
```python
sample.num_steps = 19
sample.duration = 10
```
#### Samples Per Batch
This parameter defines the number of samples per batch. It is recommended to have a larger value if available, as it can improve training efficiency.
```python
sample.num_batches_per_epoch = 32
```
#### Gradient Accumulation Steps
This setting allows for accumulating gradients across steps. It is advisable to have a larger value if available, to optimize the training process.
```python
train.gradient_accumulation_steps = 1
```

#### Change reward function
```python
config.reward_fn = "clap_score"
# or below
config.reward_fn = "emo_score"
```

<br/><br/>

### training cmd
After finish the settings, run `accelerate launch scripts/train.py` to train the model.
<br/><br/>

## Inference
After trained your model, the lora weight is stored in `logs/` directory
place your lora ckpt directory (which may include 4 files: optimizer.bin, pytorch_lora_weights.safetensors, random_states_0.pkl and scaler.pt) in `audioldm2_ckpt` directory. 

And then, change `model_path`, `output_dir`, and the setting of your diffusion model `num_inference_steps` and `audio_length_in_s` in `inference.py`. run `inference.py` to get results wavs in `output_dir`
<br/><br/>

## Evaluation
As finishing inference, to evaluate the clap similarity score or emopia accuracy, you can run `clap_eval.py` and `emopia_eval.py`. Please aware to modify the `folder_name` and `input_dir` for your result directory.

The score will store in a csv file in your directory and also print the average score in terminal.