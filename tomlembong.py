import streamlit as st
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np
import pandas as pd

# Custom CSS untuk tema sederhana dan elegan
st.markdown("""
<style>
    /* Background dan tema utama */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Dedication section - simple and clean */
    .dedication {
        background: #ffffff;
        border-left: 4px solid #3498db;
        padding: 1rem 1.5rem;
        margin: 1.5rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .dedication-text {
        color: #34495e;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Input styling */
    .stTextArea > div > div > textarea {
        border-radius: 8px !important;
        border: 2px solid #e1e8ed !important;
        font-size: 1rem !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 0 1px rgba(52, 152, 219, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3498db !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #2980b9 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Results section */
    .emotion-result {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        border-top: 4px solid #3498db;
    }
    
    .predicted-emotion {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Confidence bars */
    .confidence-item {
        background: #f8f9fa;
        margin: 0.5rem 0;
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 3px solid #3498db;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #ffffff;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f1f2f6 !important;
        border-radius: 6px !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d5f4e6 !important;
        color: #2d8659 !important;
        border: 1px solid #81c784 !important;
        border-radius: 6px !important;
    }
    
    .stError {
        background-color: #ffebee !important;
        color: #c62828 !important;
        border: 1px solid #e57373 !important;
        border-radius: 6px !important;
    }
    
    /* Clean spacing */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Example buttons */
    .example-btn .stButton > button {
        background-color: #95a5a6 !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 1rem !important;
        margin: 0.2rem !important;
    }
    
    .example-btn .stButton > button:hover {
        background-color: #7f8c8d !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model_and_tokenizer():
    """Load model dan tokenizer dengan caching"""
    try:
        model = TFAutoModelForSequenceClassification.from_pretrained("MonyetttRindam/emotion_classification_model")
        tokenizer = AutoTokenizer.from_pretrained("MonyetttRindam/emotion_classification_model")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Fungsi prediksi emosi
def predict_emotion(text, model, tokenizer):
    """Prediksi emosi dari teks input"""
    try:
        if not text or len(text.strip()) == 0:
            return None, None
        
        inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=96)
        outputs = model(**inputs)
        predictions = outputs.logits
        probabilities = tf.nn.softmax(predictions, axis=-1)
        predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]
        
        emotions = ["SADNESS", "ANGER", "SUPPORT", "HOPE", "DISAPPOINTMENT"]
        probabilities_np = probabilities[0].numpy()
        
        return emotions[predicted_class], probabilities_np
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Header
st.markdown('<h1 class="main-header">🎭 Analisis Emosi Teks</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Memahami perasaan di balik kata-kata</p>', unsafe_allow_html=True)

# Simple dedication message
st.markdown("""
<div class="dedication">
    <div class="dedication-text">
        <strong>Didedikasikan untuk mereka yang berjuang mencari kebenaran dan keadilan.</strong><br>
        "Keadilan mungkin tertunda, tetapi tidak akan pernah lenyap."
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Memuat model..."):
    model, tokenizer = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.error("Gagal memuat model. Periksa path model dan coba lagi.")
    st.stop()

st.success("✅ Model berhasil dimuat!")

# Input section
st.subheader("📝 Masukkan Teks untuk Analisis")
user_input = st.text_area(
    "Tulis teks di sini:",
    placeholder="Ketik teks yang ingin Anda analisis emosinya...",
    height=100
)

# Predict button
if st.button("🔍 Analisis Emosi", type="primary"):
    if user_input and user_input.strip():
        with st.spinner("Menganalisis..."):
            prediction, probabilities = predict_emotion(user_input, model, tokenizer)
            
            if prediction is not None:
                # Results section
                st.markdown('<div class="emotion-result">', unsafe_allow_html=True)
                
                # Main prediction
                emotion_emojis = {
                    "SADNESS": "😢", "ANGER": "😠", "SUPPORT": "🤝", 
                    "HOPE": "🌟", "DISAPPOINTMENT": "😔"
                }
                
                st.markdown(f"""
                <div class="predicted-emotion">
                    {emotion_emojis.get(prediction, "💭")} Emosi Dominan: {prediction}
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence scores
                st.subheader("📊 Tingkat Keyakinan")
                emotions = ["SADNESS", "ANGER", "SUPPORT", "HOPE", "DISAPPOINTMENT"]
                
                # Create dataframe for chart
                scores_df = pd.DataFrame({
                    'Emotion': emotions,
                    'Confidence': probabilities
                })
                scores_df = scores_df.sort_values('Confidence', ascending=False)
                
                # Simple bar chart
                st.bar_chart(data=scores_df.set_index('Emotion')['Confidence'])
                
                # Confidence details
                st.subheader("📋 Detail Skor")
                for emotion, score in zip(emotions, probabilities):
                    emoji = emotion_emojis.get(emotion, "💭")
                    st.markdown(f"""
                    <div class="confidence-item">
                        <strong>{emoji} {emotion}</strong>: {score:.4f} ({score*100:.2f}%)
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Gagal menganalisis emosi. Silakan coba lagi.")
    else:
        st.warning("Mohon masukkan teks untuk dianalisis.")

# About section
with st.expander("ℹ️ Tentang Model"):
    st.markdown("""
    **Model Analisis Emosi Berbasis BERT**
    
    Sistem ini menggunakan IndoBERT yang di-fine-tune untuk mengklasifikasi 5 emosi:
    - 😢 **SADNESS**: Kesedihan, duka
    - 😠 **ANGER**: Kemarahan, frustrasi  
    - 🤝 **SUPPORT**: Dukungan, solidaritas
    - 🌟 **HOPE**: Harapan, optimisme
    - 😔 **DISAPPOINTMENT**: Kekecewaan
    
    Model menggunakan maksimal 96 token per teks.
    """)

# Examples
with st.expander("📝 Contoh Teks"):
    example_texts = [
        "Hari ini rasanya berat, semua tidak berjalan sesuai harapan.",
        "Sangat kesal dengan situasi yang tidak adil ini!",
        "Saya yakin kebenaran akan terbukti, mari kita dukung proses yang adil.",
        "Meski sulit sekarang, saya optimis masa depan akan lebih baik.",
        "Mengecewakan melihat bagaimana sistem memperlakukan orang baik."
    ]
    
    st.markdown('<div class="example-btn">', unsafe_allow_html=True)
    cols = st.columns(2)
    
    for i, example in enumerate(example_texts):
        col = cols[i % 2]
        with col:
            if st.button(f"Contoh {i+1}", key=f"example_{i}"):
                st.session_state.example_text = example

    st.markdown('</div>', unsafe_allow_html=True)
    
    if 'example_text' in st.session_state:
        st.text_area("Contoh terpilih:", st.session_state.example_text, key="example_display")

# Simple footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">Sistem Analisis Emosi - Dibuat dengan harapan akan pemahaman yang lebih baik</p>', 
    unsafe_allow_html=True
)
