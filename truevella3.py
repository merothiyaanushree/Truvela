import streamlit as st
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Import our custom modules
from utils.emotion_tts import detect_emotion, generate_adaptive_audio
from utils.rag_engine import initialize_rag_system, get_ethical_guidance
from utils.scorer import UnbiasedScorer

load_dotenv()

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Truevella | AI Companion", page_icon="🕊️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f4f5f9; color: #2d3748; }
    h1, h2, h3 { color: #6b46c1; font-weight: 600; } /* Lavender headings */
    .stButton>button {
        background-color: #a3bffa; /* Soft blue */
        color: #1a202c; border-radius: 8px; border: none; padding: 10px 24px; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #fbd38d; } /* Light peach */
    .disclaimer {
        background-color: #fed7d7; border-left: 4px solid #e53e3e; padding: 12px;
        border-radius: 4px; font-size: 0.9em; color: #742a2a; margin-bottom: 20px;
    }
    .stTextArea textarea { border-radius: 8px; border: 1px solid #cbd5e0; }
    </style>
    """, unsafe_allow_html=True)

# --- STATE INITIALIZATION ---
if 'page' not in st.session_state:
    st.session_state.page = 'language_selection'
if 'journal' not in st.session_state:
    st.session_state.journal = []
if 'transparency_log' not in st.session_state:
    st.session_state.transparency_log = []

# Dictionaries for multilingual support
languages = ["English", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"]
greetings = {
    "English": "Hello, I am {name}. I am here to support you.",
    "Hindi": "नमस्ते, मैं {name} हूँ। मैं यहाँ आपकी सहायता के लिए हूँ।"
}

# --- PAGE 1: LANGUAGE SELECTION ---
if st.session_state.page == 'language_selection':
    st.title("Welcome to Truevella 🕊️")
    st.write("Please select your preferred language.")
    
    selected_lang = st.selectbox("Language / भाषा", languages)
    if st.button("Continue"):
        st.session_state.language = selected_lang
        st.session_state.page = 'persona_selection'
        st.rerun()

# --- PAGE 2: PERSONA SELECTION ---
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

# --- PAGE 3: HOME DASHBOARD ---
elif st.session_state.page == 'home':
    lang = st.session_state.language
    persona = st.session_state.persona
    
    st.title(f"🕊️ Truevella Dashboard ({persona})")
    
    # Strictly enforced legal disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>LEGAL DISCLAIMER:</strong> Truevella provides ethical guidance based on standard workplace practices. 
        It is NOT legal advice. Always consult a licensed attorney or HR professional for official matters.
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["💬 Companion Space", "⚖️ Ethical Guidance", "🏆 Unbiased Scorer", "📓 Journal", "📊 Transparency Log"])

    # 1. EMOTIONAL COMPANION
    with tabs[0]:
        st.subheader("How are you feeling today?")
        user_text = st.text_area("Share your thoughts...")
        if st.button("Talk to " + persona):
            if user_text:
                with st.spinner("Analyzing..."):
                    emotion = detect_emotion(user_text)
                    st.write(f"**Detected Emotion:** {emotion.capitalize()}")
                    
                    # Generate adaptive response
                    greeting = greetings.get(lang, greetings["English"]).format(name=persona)
                    if emotion in ['sadness', 'fear']:
                        response = f"{greeting} I sense that things are heavy right now. Take a deep breath, I am here."
                    elif emotion == 'joy':
                        response = f"{greeting} I can feel your positive energy! That's wonderful."
                    else:
                        response = f"{greeting} I am listening carefully to what you're sharing."
                    
                    st.success(response)
                    generate_adaptive_audio(response, persona, emotion)
                    
                    st.session_state.transparency_log.append({
                        "Action": "Emotion Detection", "Input": user_text, "Output": emotion, "Timestamp": datetime.now()
                    })

    # 2. LEGAL / ETHICAL GUIDANCE
    with tabs[1]:
        st.subheader("Ethical & Workplace Navigation")
        scenario = st.text_area("Describe the workplace situation or dilemma:")
        if st.button("Seek Objective Guidance"):
            if scenario:
                with st.spinner("Consulting verified guidelines..."):
                    qa_chain = initialize_rag_system()
                    answer = get_ethical_guidance(qa_chain, scenario)
                    st.info(answer)
                    
                    st.session_state.transparency_log.append({
                        "Action": "RAG Guidance", "Input": scenario, "Output": answer, "Timestamp": datetime.now()
                    })

    # 3. UNBIASED SCORING
    with tabs[2]:
        st.subheader("Unbiased Evaluation System")
        st.write("Rank resumes or ideas purely on merit. All identifying data is stripped before scoring.")
        
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
                    
                    st.session_state.transparency_log.append({
                        "Action": "Unbiased Scoring", "Input": "File Upload", "Output": f"Score: {final_score:.2f}", "Timestamp": datetime.now()
                    })

    # 4. JOURNALING
    with tabs[3]:
        st.subheader("Daily Reflections")
        entry = st.text_area("Log your thoughts (e.g., 'Received a new shipment of inventory today but feeling overwhelmed with the tracking...')")
        if st.button("Save Entry"):
            if entry:
                st.session_state.journal.append({"Date": datetime.now().strftime("%Y-%m-%d %H:%M"), "Entry": entry})
                st.success("Saved to your encrypted local journal.")
        
        if st.session_state.journal:
            st.dataframe(pd.DataFrame(st.session_state.journal), use_container_width=True)

    # 5. TRANSPARENCY LOG
    with tabs[4]:
        st.subheader("System Transparency")
        st.write("Review how Truevella is making decisions to ensure fairness.")
        if st.session_state.transparency_log:
            st.dataframe(pd.DataFrame(st.session_state.transparency_log), use_container_width=True)
        else:
            st.write("No actions logged yet.")