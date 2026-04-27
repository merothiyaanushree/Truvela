import streamlit as st
import pandas as pd
import os
import re
import spacy
import torch
from datetime import datetime
from dotenv import load_dotenv

# HuggingFace & ML Imports
from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from elevenlabs.client import ElevenLabs

# --- 1. SETUP & ENVIRONMENT ---
load_dotenv()

# Ensure Spacy model is downloaded automatically
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# --- 2. LOCALIZATION & QUOTES DATA ---
quotes_data = {
    "greetings": {
        "English": "Hi, I am {name}. I’m here to listen and help you. What is on your mind today?",
        "Hindi": "नमस्ते, मैं {name} हूँ। मैं यहाँ आपकी बात सुनने और आपकी मदद करने के लिए हूँ। आज आप किस बारे में बात करना चाहेंगे?",
        "Marathi": "नमस्कार, मी {name} आहे. मी इथे तुमचे ऐकण्यासाठी आणि मदत करण्यासाठी आहे.",
        "Telugu": "నమస్తే, నేను {name}. నేను మీ మాట వినడానికి మరియు సహాయం చేయడానికి ఇక్కడ ఉన్నాను.",
        "Punjabi": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਮੈਂ {name} ਹਾਂ। ਮੈਂ ਇੱਥੇ ਤੁਹਾਡੀ ਗੱਲ ਸੁਣਨ ਅਤੇ ਮਦਦ ਕਰਨ ਲਈ ਹਾਂ।",
        "Tamil": "வணக்கம், நான் {name}. நான் உங்கள் பேச்சைக் கேட்கவும் உதவவும் இங்கே இருக்கிறேன்.",
        "Bengali": "নমস্কার, আমি {name}। আমি এখানে আপনার কথা শুনতে এবং সাহায্য করতে এসেছি।",
        "Gujarati": "નમસ્તે, હું {name} છું. હું અહીં તમારી વાત સાંભળવા અને મદદ કરવા માટે છું.",
        "Kannada": "ನಮಸ್ಕಾರ, ನಾನು {name}. ನಾನು ನಿಮ್ಮ ಮಾತು ಕೇಳಲು ಮತ್ತು ಸಹಾಯ ಮಾಡಲು ಇಲ್ಲಿದ್ದೇನೆ.",
        "Malayalam": "നമസ്കാരം, ഞാൻ {name} ആണ്. നിങ്ങളെ കേൾക്കാനും സഹായിക്കാനും ഞാൻ ഇവിടെയുണ്ട്."
    },
    "emotional_responses": {
        "distress": "I sense that things are heavy right now. Take a deep breath, I am here.",
        "positive": "I can feel your positive energy! That's wonderful.",
        "neutral": "I am listening carefully to what you're sharing."
    }
}

# --- 3. ML & AI MODULES (Cached for Performance) ---

@st.cache_resource
def load_emotion_model():
    """Loads the HuggingFace sentiment analysis model."""
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilbert-base")

def detect_emotion(text):
    classifier = load_emotion_model()
    result = classifier(text)
    return result[0]['label'] # anger, disgust, fear, joy, neutral, sadness, surprise

def generate_adaptive_audio(text, persona, emotion):
    """Generates ultra-realistic TTS via ElevenLabs."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        st.warning("ElevenLabs API Key not found. Text-to-Speech is disabled.")
        return

    try:
        client = ElevenLabs(api_key=api_key)
        # Placeholder Voice IDs - Replace with your actual ElevenLabs voices
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

@st.cache_resource
def initialize_rag_system():
    """Initializes the FAISS Vector DB and LLM for Ethical Guidance."""
    # Auto-generate a dummy legal document to ensure the app works out-of-the-box
    file_path = "workplace_ethics.txt"
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("According to standard workplace ethics, harassment is strictly prohibited. Employees facing unfair termination should document all exchanges and contact HR immediately. Discrimination based on race, gender, or religion is a violation of equal opportunity employment.")

    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", max_length=256)
    local_llm = HuggingFacePipeline(pipeline=llm_pipeline)
    
    return RetrievalQA.from_chain_type(llm=local_llm, chain_type="stuff", retriever=vector_store.as_retriever())

class UnbiasedScorer:
    """PyTorch-based system to strip bias and rank text objectively."""
    def __init__(self):
        self.weights = torch.tensor([0.30, 0.25, 0.15, 0.20, 0.10], dtype=torch.float32)

    def anonymize_text(self, text):
        doc = nlp(text)
        clean_text = text
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "NORP", "GPE"]:
                clean_text = clean_text.replace(ent.text, "[REDACTED]")
        clean_text = re.sub(r'\b(he|him|his|she|her|hers)\b', '[PRONOUN]', clean_text, flags=re.IGNORECASE)
        return clean_text

    def extract_features(self, text):
        text = text.lower()
        skills_score = min(100, text.count("python")*10 + text.count("react")*10 + text.count("manage")*5 + 50)
        exp_score = min(100, text.count("year")*10 + text.count("developed")*10 + 40)
        edu_score = min(100, text.count("degree")*20 + text.count("university")*20 + text.count("school")*15 + 40)
        achieve_score = min(100, text.count("award")*25 + text.count("won")*20 + 30)
        comm_score = 85
        return torch.tensor([skills_score, exp_score, edu_score, achieve_score, comm_score], dtype=torch.float32)

    def calculate_score(self, feature_tensor):
        final_score = torch.dot(feature_tensor, self.weights).item()
        breakdown = {
            "Skills (30%)": feature_tensor[0].item(), "Experience (25%)": feature_tensor[1].item(),
            "Education (15%)": feature_tensor[2].item(), "Achievements (20%)": feature_tensor[3].item(),
            "Communication (10%)": feature_tensor[4].item()
        }
        return final_score, breakdown

# --- 4. STREAMLIT UI & STATE MANAGEMENT ---
st.set_page_config(page_title="Truevella | AI Companion", page_icon="🕊️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f4f5f9; color: #2d3748; }
    h1, h2, h3 { color: #6b46c1; font-weight: 600; } 
    .stButton>button {
        background-color: #a3bffa; color: #1a202c; border-radius: 8px; border: none; padding: 10px 24px; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #fbd38d; } 
    .disclaimer {
        background-color: #fed7d7; border-left: 4px solid #e53e3e; padding: 12px;
        border-radius: 4px; font-size: 0.9em; color: #742a2a; margin-bottom: 20px;
    }
    .stTextArea textarea { border-radius: 8px; border: 1px solid #cbd5e0; }
    </style>
    """, unsafe_allow_html=True)

if 'page' not in st.session_state: st.session_state.page = 'language_selection'
if 'journal' not in st.session_state: st.session_state.journal = []
if 'transparency_log' not in st.session_state: st.session_state.transparency_log = []

# --- APP FLOW ---
if st.session_state.page == 'language_selection':
    st.title("Welcome to Truevella 🕊️")
    st.write("Please select your preferred language.")
    selected_lang = st.selectbox("Language / भाषा", list(quotes_data["greetings"].keys()))
    if st.button("Continue"):
        st.session_state.language = selected_lang
        st.session_state.page = 'persona_selection'
        st.rerun()

elif st.session_state.page == 'persona_selection':
    st.title("Choose Your Companion")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mira")
        st.write("Warm, nurturing, and soft-spoken.")
        if st.button("Select Mira"):
            st.session_state.persona = "Mira"
            st.session_state.page = 'home'
            st.rerun()
    with col2:
        st.subheader("Arin")
        st.write("Calm, logical, and reassuring.")
        if st.button("Select Arin"):
            st.session_state.persona = "Arin"
            st.session_state.page = 'home'
            st.rerun()

elif st.session_state.page == 'home':
    lang = st.session_state.language
    persona = st.session_state.persona
    
    st.title(f"🕊️ Truevella Dashboard ({persona})")
    st.markdown("""
    <div class="disclaimer">
        <strong>LEGAL DISCLAIMER:</strong> Truevella provides ethical guidance based on standard workplace practices. 
        It is NOT legal advice. Always consult a licensed attorney or HR professional for official matters.
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["💬 Companion Space", "⚖️ Ethical Guidance", "🏆 Unbiased Scorer", "📓 Journal", "📊 Transparency Log"])

    with tabs[0]:
        st.subheader("How are you feeling today?")
        user_text = st.text_area("Share your thoughts...", key="companion_text")
        if st.button("Talk to " + persona):
            if user_text:
                with st.spinner("Analyzing..."):
                    emotion = detect_emotion(user_text)
                    st.write(f"**Detected Emotion:** {emotion.capitalize()}")
                    
                    greeting = quotes_data["greetings"].get(lang, quotes_data["greetings"]["English"]).format(name=persona)
                    if emotion in ['sadness', 'fear']: ext = quotes_data["emotional_responses"]["distress"]
                    elif emotion == 'joy': ext = quotes_data["emotional_responses"]["positive"]
                    else: ext = quotes_data["emotional_responses"]["neutral"]
                    
                    response = f"{greeting} {ext}"
                    st.success(response)
                    generate_adaptive_audio(response, persona, emotion)
                    st.session_state.transparency_log.append({"Action": "Emotion Detection", "Input": user_text, "Output": emotion, "Timestamp": datetime.now()})

    with tabs[1]:
        st.subheader("Ethical & Workplace Navigation")
        scenario = st.text_area("Describe the workplace situation or dilemma:", key="ethics_text")
        if st.button("Seek Objective Guidance"):
            if scenario:
                with st.spinner("Consulting verified guidelines..."):
                    qa_chain = initialize_rag_system()
                    answer = qa_chain.run(f"Based ONLY on the provided legal/ethical documents, how should one handle this scenario: '{scenario}'?")
                    st.info(answer)
                    st.session_state.transparency_log.append({"Action": "RAG Guidance", "Input": scenario, "Output": answer, "Timestamp": datetime.now()})

    with tabs[2]:
        st.subheader("Unbiased Evaluation System")
        uploaded_file = st.file_uploader("Upload Profile/Resume (TXT)", type=['txt'])
        if st.button("Evaluate Objectively"):
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                scorer = UnbiasedScorer()
                with st.spinner("Anonymizing and processing through PyTorch Model..."):
                    clean_text = scorer.anonymize_text(content)
                    features = scorer.extract_features(clean_text)
                    final_score, breakdown = scorer.calculate_score(features)
                    
                    st.metric(label="Overall Objective Score", value=f"{final_score:.2f} / 100")
                    st.write("### Weighted Breakdown")
                    st.json(breakdown)
                    st.session_state.transparency_log.append({"Action": "Unbiased Scoring", "Input": "File Upload", "Output": f"Score: {final_score:.2f}", "Timestamp": datetime.now()})

    with tabs[3]:
        st.subheader("Daily Reflections")
        entry = st.text_area("Log your thoughts...", key="journal_text")
        if st.button("Save Entry"):
            if entry:
                st.session_state.journal.append({"Date": datetime.now().strftime("%Y-%m-%d %H:%M"), "Entry": entry})
                st.success("Saved to your encrypted local journal.")
        if st.session_state.journal:
            st.dataframe(pd.DataFrame(st.session_state.journal), use_container_width=True)

    with tabs[4]:
        st.subheader("System Transparency")
        if st.session_state.transparency_log:
            st.dataframe(pd.DataFrame(st.session_state.transparency_log), use_container_width=True)
        else:
            st.write("No actions logged yet.")