import streamlit as st

st.set_page_config(page_title="TruVella AI", layout="centered")

st.title("🌿 TruVella AI Assistant")

st.write("An AI-powered assistant for guidance, resume insights, and emotional support.")

# Language selection
language = st.selectbox("Choose your language", ["English", "Hindi"])

# User input
user_input = st.text_area("Enter your text / query")

# Basic AI response simulation
if user_input:
    if "sad" in user_input.lower():
        response = "I'm here for you. Things will get better. 💙"
    elif "job" in user_input.lower():
        response = "Focus on skills, clarity, and achievements in your resume."
    elif "law" in user_input.lower():
        response = "This is general legal information. Please consult a professional for advice."
    else:
        response = "Thank you for sharing. TruVella is here to guide you."

    st.success(response)

# Resume scoring demo
st.subheader("📄 Resume Score (Demo)")
resume_text = st.text_area("Paste your resume content")

if resume_text:
    score = min(len(resume_text) // 50, 100)
    st.write(f"Your resume score: {score}/100")

st.caption("⚠️ This is a prototype version of TruVella AI.")