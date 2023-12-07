from PIL import Image
import io
import numpy as np
import torch

from CLAP.src import laion_clap
from numpy.linalg import norm
import torchaudio

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def clap_score():

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    model.load_ckpt('/data/fundwotsai/RL_final/ddpo-pytorch/CLAP/ckpt/music_audioset_epoch_15_esc_90.14.pt')
    model.eval()
    model.to("cpu")
    def _fn(audios, prompts, metadata):

        if isinstance(audios, torch.Tensor):
            audios = audios.cpu().numpy() # [B, T]
            if len(audios.shape) == 3:
                audios = audios.squeeze(1) # [B, 1, T] -> [B, T]

        waveform_tensor = torch.tensor(audios)
        # waveform_tensor = waveform_tensor.to(torch.float32)
        # Save the tensor as a WAV file
        # print("in reward",prompts)
        audio_embed = model.get_audio_embedding_from_data(x = audios, use_tensor=False)
        text_embed = model.get_text_embedding(prompts)
        audio_norm = audio_embed / np.linalg.norm(audio_embed)
        text_norm = text_embed / np.linalg.norm(text_embed)
        
        # Compute the cosine similarity
        similarity = np.dot(audio_norm, text_norm.T)
        # print(similarity)
        torchaudio.save(f"{prompts}_{similarity}.wav", waveform_tensor, 16000)
        # print(waveform_tensor)
        return np.array(similarity),{}
    return _fn


def loud_score():
    
    def _fn(audios, prompts, metadata):

        if isinstance(audios, torch.Tensor):
            audios = audios.cpu().numpy() # [B, T]
            if len(audios.shape) == 3:
                audios = audios.squeeze(1) # [B, 1, T] -> [B, T]

        loudness = audios.mean(axis=-1)
        # torchaudio.save(f"{prompts}_{similarity}.wav", waveform_tensor, 16000)
        # print(waveform_tensor)
        return np.array(loudness),{}
    return _fn


def emo_score():
    from .emo_scorer import predict
    
    prompt_emo_dict = {}
    with open("/data/fundwotsai/RL_final/ddpo-pytorch/ddpo_pytorch/assets/categories.txt", "r") as f:
        for line in f.readlines():
            key, value = line.split(", ")
            prompt_emo_dict[key] = value
    

    def _fn(audios, prompts, metadata):

        if isinstance(audios, torch.Tensor):
            audios = audios.cpu().numpy() # [B, T]
            if len(audios.shape) == 3:
                audios = audios.squeeze(1) # [B, 1, T] -> [B, T]

        reward = []
        for audio, prompt in zip(audios, prompts): #[B, T]
            # by label
            # emo = predict(audio) # [T] -> str
            # if prompt_emo_dict[prompt.split(" ")[0]] == emo:
            #     reward.append(1)
            # else:
            #     reward.append(0)
            
            # by value
            emo_value = predict(audio) # [T] -> str
            idx = int(prompt_emo_dict[prompt.split(" ")[0]][1]) - 1
            reward.append(emo_value[idx])
        
        return np.array(reward)[None],{}
    return _fn
