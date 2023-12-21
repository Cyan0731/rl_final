# rl_final


## Preparation

after git clone this repo, cd to `ddpo_pytorch` directory and `pip install -e .` to install the repo.

download CLAP weight below and move CLAP weight into `ddpo-pytorch/CLAP/ckpt` directory
https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt
<br/><br/>

## Training

(set param in base.py) 

(change prompt in ddpo_pytorch/assests/)

After finish the settings, run `accelerate launch scripts/train.py` to train the model.


<br/><br/>

## Inference
After trained your model, the lora weight is stored in `logs/` directory
place your lora ckpt directory (which may include 4 files: optimizer.bin, pytorch_lora_weights.safetensors, random_states_0.pkl and scaler.pt) in the `audioldm2_ckpt`. 

And then, edit the `model_path`, `output_dir`, and the setting of your diffusion model `num_inference_steps` and `audio_length_in_s` in `inference.py`. run `inference.py` to get results wavs in `output_dir`
<br/><br/>

## Evaluation
As finishing inference, to evaluate the clap similarity score or emopia accuracy, you can run `clap_eval.py` and `emopia_eval.py`. Please aware to modify the `folder_name` and `input_dir` for your result directory.

The score will store in a csv file in your directory and also print the average score in terminal.