from scripts.pipeline_audioldm2 import AudioLDM2Pipeline
import scipy
import os


# model_path = '/home/cyan/rl_final/ddpo-pytorch/logs/4emo_piano_steps19_emopia/checkpoints/checkpoint_4'
model_path = "audioldm2_ckpt/4emo_steps38_highest/checkpoint_14"
output_dir = '../samples/step38_sec10_emo_gen_highest/'
num_inference_steps = 38
audio_length_in_s = 5.


if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

pretrained_model_name_or_path = "audioldm2-music"

audioldmpipeline= AudioLDM2Pipeline.from_pretrained(
        pretrained_model_name_or_path,
    )

audioldmpipeline.unet.load_attn_procs(model_path)
audioldmpipeline.to("cuda")


emos = ['happy', 'angry', 'sad', 'tender']
# emos = ['happy', 'mad', 'sorrow', 'exciting']
emo_dict = {'happy': 'Q1', 'angry':'Q2', 'sad':'Q3', 'tender':'Q4'}
# emo_dict = {'happy': 'Q1', 'mad':'Q2', 'sorrow':'Q3', 'exciting':'Q4'}

for emo in emos:
    prompts = []
    for i , prompt in enumerate([f'a recording of an {emo} piano solo']*25):
        prompts.append(prompt)
    

    # audio = audioldmpipeline(prompt, num_inference_steps=19, audio_length_in_s=10.0).audios[0]
    audios = audioldmpipeline(prompts, negative_prompt = ["Low quality, mutiple sound source"]*25, eta=1., num_inference_steps=19, audio_length_in_s=10.0).audios
    for i in range(25):
        scipy.io.wavfile.write(os.path.join(output_dir, f"{emo_dict[emo]}_{i}.wav"), rate=16000, data=audios[i])
