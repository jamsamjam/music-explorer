import librosa
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load("audio.wav")

hop_length = 512
frame_length = 2048
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

frames = range(len(rms))
t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

plt.figure(figsize=(10, 4))
plt.plot(t, rms, label='RMS Energy')
plt.title('Dynamic Build-Up in Time (RMS Energy Over Time)')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.legend()
plt.show()
