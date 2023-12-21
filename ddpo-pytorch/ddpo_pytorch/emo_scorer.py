import os
import json
import pickle
from pathlib import Path
from argparse import ArgumentParser, Namespace
import numpy as np

import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
import librosa
from audio_cls.src.model.net import ShortChunkCNN_Res


def predict(audio):
    device = "cpu"
    config_path = Path("/home/cyan/rl_final/ddpo-pytorch/ddpo_pytorch/emo_weight/ar_va/hparams.yaml")
    checkpoint_path = Path("/home/cyan/rl_final/ddpo-pytorch/ddpo_pytorch/emo_weight/ar_va/best.ckpt")
    config = OmegaConf.load(config_path)
    label_list = list(config.task.labels)

    model = ShortChunkCNN_Res(
            sample_rate = config.wav.sr,
            n_fft = config.hparams.n_fft,
            f_min = config.hparams.f_min,
            f_max = config.hparams.f_max,
            n_mels = config.hparams.n_mels,
            n_channels = config.hparams.n_channels,
            n_class = config.task.n_class
    )

    state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
    new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
    new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.to(device)
    
    audio = librosa.resample(audio, orig_sr=16000, target_sr=22050)
    audio = torch.tensor(audio)[None]
    model_input = audio.mean(0, True) # [2, T] -> T

    sample_length = config.wav.sr * config.wav.input_length
    frame = ((model_input.shape[1] - sample_length) // sample_length) + 1
    audio_sample = torch.zeros(frame, 1, sample_length)
    for i in range(frame):
        audio_sample[i] = torch.Tensor(model_input[:,i*sample_length:(i+1)*sample_length])
    with torch.no_grad():
        prediction = model(audio_sample.to(device))
        prediction = prediction.mean(0,False)
    
    pred_label = label_list[prediction.squeeze(0).max(0)[1].detach().cpu().numpy()]
    pred_value = prediction.squeeze(0).detach().cpu().numpy()

    return pred_label, pred_value

if __name__ == "__main__":
    # test case
    audio = np.array([0.]*16000*5) # [T]
    pred_label = predict(audio)
    print(pred_label)
