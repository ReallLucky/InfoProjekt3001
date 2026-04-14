import streamlit as st
import tempfile

from pose_analysis import extract_pose_sequence, compute_features, classify_strike
from feedback import calculate_score, generate_feedback

st.set_page_config(page_title="Eskrima Trainer", layout="centered")

st.title("🥋 Eskrima Schlaganalyse (KI Trainer)")
st.write("Lade ein Video deines Schlages hoch und erhalte Feedback.")

# Modus Auswahl
mode = st.selectbox(
    "Analyse-Modus wählen:",
    ["Vortrainierte KI (empfohlen)", "Teachable Machine (optional)"]
)

uploaded_file = st.file_uploader("Video hochladen", type=["mp4", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(uploaded_file)

    if st.button("Analyse starten"):
        st.write("⏳ Analysiere Video...")

        sequence = extract_pose_sequence(tfile.name)

        if len(sequence) < 5:
            st.error("Zu wenig Bewegung erkannt. Bitte neues Video aufnehmen.")
        else:
            features = compute_features(sequence)

            if mode == "Vortrainierte KI (empfohlen)":
                strike = classify_strike(features)
            else:
                strike = "Teachable Machine Modell noch nicht integriert"

            score = calculate_score(features)
            feedback = generate_feedback(score, features)

            st.subheader("Ergebnis")
            st.write(f"**Erkannter Schlag:** {strike}")
            st.write(f"**Score:** {score}/100")

            st.subheader("Feedback")
            for f in feedback:
                st.write(f"- {f}")