import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Music Feature Explorer", layout="wide")
st.title("🎼 Structural Music Feature Explorer")
st.markdown("Upload a music file to explore its dynamic, rhythmic, and temporal structure through visual analysis.")

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file:

    y, sr = librosa.load(uploaded_file)

    # Prepare features
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    t_rms = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=512)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    t_onset = librosa.times_like(onset_env, sr=sr)

    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    intervals = np.diff(beat_times)

    fig1, ax1 = plt.subplots(figsize=(3,2))
    ax1.plot(t_rms, rms, color='crimson', label='RMS Energy', lw=0.6)
    ax1.set_xlabel("Time (s)", fontsize=9)
    ax1.set_ylabel("Energy", fontsize=9)
    ax1.set_title("RMS Energy", fontsize=10)
    ax1.legend(fontsize=7)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(3,2))
    ax2.plot(t_onset, onset_env, color='darkgreen', label='Onset Strength', lw=0.6)
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.set_ylabel("Strength", fontsize=9)
    ax2.set_title("Onset Strength", fontsize=10)
    ax2.legend(fontsize=7)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(3,2))
    ax3.plot(intervals, color='navy', label='Inter-beat Interval (s)', lw=0.6)
    ax3.set_xlabel("Beat Index", fontsize=9)
    ax3.set_ylabel("Time Between Beats (s)", fontsize=9)
    ax3.set_title("Tempo Variability", fontsize=10)
    ax3.legend(fontsize=7)
    fig3.tight_layout()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**1. 📈 Dynamic Profile: RMS Energy**")
        st.pyplot(fig1)
        st.markdown("""
        <span style='font-size:15px;'>
        RMS (Root Mean Square) energy shows how loud or intense the sound is over time.<br>
        For example, in <i>Hans Zimmer - Time</i>, this reflects the gradual build-up of dynamic intensity that gives the piece its emotional force.
        </span>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**2. 🥁 Rhythmic Activity: Onset Strength**")
        st.pyplot(fig2)
        st.markdown("""
        <span style='font-size:15px;'>
        Onset strength detects when new sounds or instruments enter.<br>
        Peaks in this graph often correspond to instrumental entries or emphasized moments.
        </span>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("**3. ⏱️ Temporal Flow: Tempo & Beat Intervals**")
        st.pyplot(fig3)
        st.markdown(f"""
        <span style='font-size:15px;'>
        Estimated tempo: <b>{tempo_val:.2f} BPM</b><br>
        Inter-beat intervals show the consistency (or variation) in tempo over time.<br>
        This can reveal moments of rhythmic change or expressive timing.
        </span>
        """, unsafe_allow_html=True)

else:
    st.info("Please upload a `.wav` audio file to begin the analysis.")
