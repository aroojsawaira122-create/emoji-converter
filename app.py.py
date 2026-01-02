# ==============================
# 😄 AI EMOJI CONVERTER (ADVANCED)
# Single File Version
# ==============================
# -*- coding: utf-8 -*-
import streamlit as st
from transformers import pipeline

# ==============================
# 😀 LARGE EMOJI DATABASE
# ==============================

emoji_map = {
    "joy": ["😄", "😁", "😊", "🥳", "🎉", "✨", "😆", "😃"],
    "sadness": ["😢", "😭", "💔", "😞", "😔", "🌧️", "🥀"],
    "anger": ["😡", "🤬", "🔥", "💢", "😠", "👿"],
    "fear": ["😨", "😰", "😱", "🫣", "😟"],
    "love": ["❤️", "😍", "😘", "💖", "💕", "💘", "💞"],
    "surprise": ["😮", "😲", "🤯", "😯", "🎊"],
    "neutral": ["🙂", "😐", "😶"],
    "disgust": ["🤢", "🤮", "😖"],
    "confidence": ["😎", "💪", "🔥"],
    "excited": ["🤩", "🚀", "🎉", "🔥"],
}

def get_emojis(emotion):
    return " ".join(emoji_map.get(emotion, emoji_map["neutral"]))

# ==============================
# 🧠 AI EMOTION DETECTION MODEL
# ==============================

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_model = load_model()

def detect_emotion(text):
    results = emotion_model(text)[0]
    best_emotion = max(results, key=lambda x: x["score"])
    return best_emotion["label"]

def emoji_converter(text):
    emotion = detect_emotion(text)
    emojis = get_emojis(emotion)
    return emotion.upper(), emojis

# ==============================
# 🌐 STREAMLIT WEB INTERFACE
# ==============================

st.set_page_config(
    page_title="AI Emoji Converter",
    page_icon="😄",
    layout="centered"
)

st.title("😄 AI Emoji Converter")
st.write("🧠 **AI text ko samajh kar perfect emojis deta hai**")

text = st.text_area(
    "✍️ Apna sentence likho:",
    placeholder="Example: I am extremely happy today!",
    height=120
)

if st.button("🚀 Convert to Emojis"):
    if text.strip() == "":
        st.warning("⚠️ Please koi sentence likho")
    else:
        with st.spinner("🔍 Emotion detect ho raha hai..."):
            emotion, emojis = emoji_converter(text)

        st.success("✅ Conversion Complete!")
        st.markdown(f"### 🧠 Emotion Detected: **{emotion}**")
        st.markdown(f"### 😀 Emojis: {emojis}")

st.markdown("---")
st.caption("🔬 Powered by Hugging Face Transformers | 🎓 Advanced AI Project")
