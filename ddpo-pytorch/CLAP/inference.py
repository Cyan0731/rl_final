from src import laion_clap
from numpy import dot
from numpy.linalg import norm

model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
model.load_ckpt('ckpt/music_audioset_epoch_15_esc_90.14.pt')

audio_file = [
    'audio/funk_groove_guitar.wav',
    'audio/funk_prier.wav'
]

audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=False)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)

text_data = ["funk groove guitar", "funk prier"] 
text_embed = model.get_text_embedding(text_data)
# print(text_embed)
# print(text_embed.shape)

score = []
for i in range(len(audio_file)):
    sim = dot(audio_embed[i], text_embed[i])/ (norm(audio_embed[i])*norm(text_embed[i]))
    score.append(sim)

print(score)
# [0.20150657, 0.0685468]