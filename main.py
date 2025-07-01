import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Music Feature Explorer", layout="centered")
st.title("🎼 Structural Music Feature Explorer")
st.markdown("Upload a music file to explore its dynamic, rhythmic, and temporal structure through visual analysis.")

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file:
    y, sr = librosa.load(uploaded_file)

    # --- RMS Energy ---
    st.header("1. 📈 Dynamic Profile: RMS Energy")
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    t_rms = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=512)

    fig1, ax1 = plt.subplots()
    ax1.plot(t_rms, rms, color='crimson', label='RMS Energy')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Energy")
    ax1.set_title("RMS Energy Over Time")
    ax1.legend()
    st.pyplot(fig1)

    st.markdown("""
    **Interpretation:**  
    RMS (Root Mean Square) energy shows how loud or intense the sound is over time.  
    In *Time*, this reflects the gradual build-up of dynamic intensity that gives the piece its emotional force.
    """)

    # --- Onset Strength ---
    st.header("2. 🥁 Rhythmic Activity: Onset Strength")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    t_onset = librosa.times_like(onset_env, sr=sr)

    fig2, ax2 = plt.subplots()
    ax2.plot(t_onset, onset_env, color='darkgreen', label='Onset Strength')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Strength")
    ax2.set_title("Onset Strength Curve")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("""
    **Interpretation:**  
    Onset strength detects when new sounds or instruments enter.  
    Peaks in this graph often correspond to instrumental entries or emphasized moments.
    """)

    # --- Tempo / Inter-beat Interval ---
    st.header("3. ⏱️ Temporal Flow: Tempo & Beat Intervals")
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    intervals = np.diff(beat_times)

    fig3, ax3 = plt.subplots()
    ax3.plot(intervals, color='navy', label='Inter-beat Interval (s)')
    ax3.set_xlabel("Beat Index")
    ax3.set_ylabel("Time Between Beats (s)")
    ax3.set_title("Tempo Variability")
    ax3.legend()
    st.pyplot(fig3)

    st.markdown(f"""
    **Interpretation:**  
    Estimated tempo: **{tempo_val:.2f} BPM**  
    Inter-beat intervals show the consistency (or variation) in tempo over time.  
    This can reveal moments of rhythmic change or expressive timing.
    """)

else:
    st.info("Please upload a `.wav` audio file to begin the analysis.")
