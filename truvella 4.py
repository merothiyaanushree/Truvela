import os
import streamlit as st
from transformers import pipeline
from elevenlabs.client import ElevenLabs

# Cache the ML model to prevent reloading on every interaction
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilbert-base")

def detect_emotion(text):
    classifier = load_emotion_model()
    result = classifier(text)
    return result[0]['label'] # Returns anger, disgust, fear, joy, neutral, sadness, surprise

def generate_adaptive_audio(text, persona, emotion):
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        st.warning("ElevenLabs API Key not found. Audio disabled.")
        return

    try:
        client = ElevenLabs(api_key=api_key)
        # Map persona to specific voice IDs (Replace with your actual ElevenLabs Voice IDs)
        voice_id = "EXAVITQu4vr4xnSDxMaL" if persona == "Mira" else "pNInz6obbf5pNrqzCawG"
        
        audio_generator = client.generate(
            text=text,
            voice=voice_id,
            model="eleven_multilingual_v2"
        )
        audio_bytes = b"".join(list(audio_generator))
        st.audio(audio_bytes, format='audio/mp3')
    except Exception as e:
        st.error(f"TTS Error: {e}")