import librosa
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load("audio.wav")

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
times = librosa.times_like(onset_env, sr=sr)
plt.plot(times, onset_env)
plt.title("Onset Strength Curve (Instrumental Build-Up)")
plt.xlabel("Time (s)")
plt.ylabel("Onset Strength")
plt.show()