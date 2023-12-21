import librosa
from glob import glob
import pandas as pd
import numpy as np
from pathlib import Path
import os

from emo_scorer_for_eval import predict


folder_name = "step19_sec10_gen"



input_dir = f"/home/cyan/rl_final/samples/{folder_name}"
input_files = glob(os.path.join(input_dir, '*.wav'))
input_files.sort()

df = pd.DataFrame(columns = ["path", "label", "pred"])
count = 0

for input_file in input_files:
    fname = os.path.basename(input_file).split('.')[0]
    emo = os.path.basename(input_file).split("_")[0]
    audio, sr = librosa.load(input_file, sr=16000)
    pred_label, _ = predict(audio)

    df.loc[len(df)] = [fname, emo, pred_label]
    if emo == pred_label:
        count += 1

print(count / len(input_files))
df.to_csv(f"/home/cyan/rl_final/samples/{folder_name}/emo_benchmark.csv", index=False)