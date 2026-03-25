import streamlit as st
import tempfile
import os
from src.predict import predict_video

st.title("Gebärdensprache Erkennung 🤟")
st.write("Lade ein Video hoch und das Modell erkennt das Wort.")

uploaded_file = st.file_uploader("Video hochladen", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    
    st.video(uploaded_file)

    if st.button("Run Model"):
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.write("Modell läuft...")

        # HIER später dein echtes Model einbauen
        prediction = predict_video(video_path)
        
        st.success(f"Erkanntes Wort: {prediction}")