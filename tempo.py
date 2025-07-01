import librosa
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load("audio.wav")
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

times = librosa.frames_to_time(beats, sr=sr)

plt.figure(figsize=(10, 4))
plt.plot(np.diff(times), label="Inter-beat Interval (s)")
plt.title("Perceived Tempo Variability in Time")
plt.xlabel("Beat index")
plt.ylabel("Interval (s)")
plt.legend()
plt.show()
