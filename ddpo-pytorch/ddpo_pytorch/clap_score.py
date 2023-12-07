from CLAP.src import laion_clap
import torch
import numpy as np
from numpy.linalg import norm


def clap_score():

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    model.load_ckpt('ckpt/music_audioset_epoch_15_esc_90.14.pt')
    model.eval()

    batch_size = 16

    def _fn(audios, prompts):

        if isinstance(audios, torch.Tensor):
            audios = audios.cpu().numpy() # [B, T]
            if len(audios.shape) == 3:
                audios = audios.squeeze(1) # [B, 1, T] -> [B, T]


        audios_batched = np.array_split(audios, np.ceil(len(audios) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []

        for audio_batch, prompt_batch in zip(audios_batched, prompts_batched):
            B, C, T = audio_batch.shape

            audio_embed = model.get_audio_embedding_from_data(x = audio_batch, use_tensor=False)
            text_embed = model.get_text_embedding(prompt_batch)

            for i in range(B):
                score = np.dot(audio_embed[i], text_embed[i]) / (norm(audio_embed[i])*norm(text_embed[i]))
                all_scores.append(score)

        return np.array(all_scores)

    return _fn
