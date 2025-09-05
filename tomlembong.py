import streamlit as st
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np
import pandas as pd

# Custom CSS untuk tema sederhana dan elegan dengan nuansa sedih
st.markdown("""
<style>
    /* Background dengan nuansa sedih */
    .stApp {
        background: linear-gradient(to bottom, #e6f7ff, #f0f8ff);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
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
        color: #5d6d7e;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Dedication section */
    .dedication {
        background: rgba(255, 255, 255, 0.7);
        border-left: 4px solid #3498db;
        padding: 1rem 1.5rem;
        margin: 1.5rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .dedication-text {
        color: #34495e;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Input styling */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #d6eaf8;
        font-size: 1rem;
        padding: 12px;
        background-color: rgba(255, 255, 255, 0.8);
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #2980b9;
    }
    
    /* Results section */
    .emotion-result {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        border-top: 4px solid #3498db;
    }
    
    .predicted-emotion {
        background: linear-gradient(135deg, #a8c0ff 0%, #3f2b96 100%);
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
    
    /* Example buttons */
    .example-btn .stButton > button {
        background-color: #a0aec0;
        font-size: 0.85rem;
        padding: 0.4rem 1rem;
        margin: 0.2rem;
        width: auto;
    }
    
    .example-btn .stButton > button:hover {
        background-color: #718096;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Tech badge */
    .tech-badge {
        display: inline-block;
        background-color: #3498db;
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
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
st.markdown('<p class="subtitle">Memahami perasaan di balik kata-kata dengan teknologi AI</p>', unsafe_allow_html=True)

# Badge teknologi
st.markdown("""
<div style="text-align: center; margin-bottom: 1rem;">
    <span class="tech-badge">Transformers</span>
    <span class="tech-badge">Deep Learning</span>
    <span class="tech-badge">IndoBERT</span>
</div>
""", unsafe_allow_html=True)

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
                
                emotion_labels = {
                    "SADNESS": "Kesedihan", 
                    "ANGER": "Kemarahan", 
                    "SUPPORT": "Dukungan", 
                    "HOPE": "Harapan", 
                    "DISAPPOINTMENT": "Kekecewaan"
                }
                
                st.markdown(f"""
                <div class="predicted-emotion">
                    {emotion_emojis.get(prediction, "💭")} Emosi Dominan: {emotion_labels.get(prediction, prediction)}
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence scores
                st.subheader("📊 Tingkat Keyakinan")
                emotions = ["SADNESS", "ANGER", "SUPPORT", "HOPE", "DISAPPOINTMENT"]
                emotion_labels_list = ["Kesedihan", "Kemarahan", "Dukungan", "Harapan", "Kekecewaan"]
                emotion_emojis_list = ["😢", "😠", "🤝", "🌟", "😔"]
                
                # Create dataframe for chart
                scores_df = pd.DataFrame({
                    'Emosi': emotion_labels_list,
                    'Emoji': emotion_emojis_list,
                    'Confidence': probabilities
                })
                scores_df = scores_df.sort_values('Confidence', ascending=False)
                
                # Simple bar chart
                st.bar_chart(data=scores_df.set_index('Emosi')['Confidence'])
                
                # Confidence details
                st.subheader("📋 Detail Skor")
                for emotion, label, emoji, score in zip(emotions, emotion_labels_list, emotion_emojis_list, probabilities):
                    st.markdown(f"""
                    <div class="confidence-item">
                        <strong>{emoji} {label}</strong>: {score:.4f} ({score*100:.2f}%)
                    </div>
                    """, unsafe_allow_html=True)
                
                # Kredit model
                st.markdown("""
                <div style="margin-top: 1.5rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <small>Analisis ini dilakukan menggunakan model <strong>IndoBERT</strong> yang telah dilatih khusus untuk klasifikasi emosi dalam teks bahasa Indonesia oleh <strong>MonyetttRindam</strong>.</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Gagal menganalisis emosi. Silakan coba lagi.")
    else:
        st.warning("Mohon masukkan teks untuk dianalisis.")

# About section
with st.expander("ℹ️ Tentang Model dan Teknologi"):
    st.markdown("""
    **Model Analisis Emosi Berbasis Transformer**
    
    Sistem ini menggunakan arsitektur **IndoBERT** (Indonesian Bidirectional Encoder Representations from Transformers) 
    yang telah di-fine-tune khusus untuk tugas klasifikasi emosi dalam teks bahasa Indonesia.
    
    ### Emosi yang Dikenali:
    - 😢 **SADNESS (Kesedihan)**: Perasaan sedih, duka, atau kepiluan
    - 😠 **ANGER (Kemarahan)**: Emosi kuat berupa kemarahan, frustrasi, atau amarah  
    - 🤝 **SUPPORT (Dukungan)**: Ekspresi dukungan, solidaritas, atau dorongan
    - 🌟 **HOPE (Harapan)**: Perasaan optimis, harapan, atau antisipasi positif
    - 😔 **DISAPPOINTMENT (Kekecewaan)**: Perasaan kecewa, tidak terpenuhi harapan
    
    ### Teknologi di Baliknya:
    - **Transformers**: Arsitektur state-of-the-art untuk pemrosesan bahasa alami
    - **Transfer Learning**: Memanfaatkan pengetahuan dari model pre-trained IndoBERT
    - **Fine-tuning**: Penyesuaian khusus untuk tugas klasifikasi emosi
    
    Model menggunakan maksimal 96 token per teks dan mampu memahami konteks serta nuansa dalam bahasa Indonesia.
    """)

# Dataset information
with st.expander("📊 Informasi Dataset"):
    st.markdown("""
    **Dataset Pelatihan Model**
    
    Model ini dilatih menggunakan kumpulan data teks bahasa Indonesia yang telah diberi label emosi secara manual.
    
    ### Karakteristik Dataset:
    - **Sumber**: Kumpulan teks dari berbagai sumber media Indonesia
    - **Jumlah sampel**: 5.000+ teks berlabel
    - **Distribusi emosi**:
        - Kesedihan (SADNESS): 25%
        - Kemarahan (ANGER): 20%
        - Dukungan (SUPPORT): 20%
        - Harapan (HOPE): 18%
        - Kekecewaan (DISAPPOINTMENT): 17%
    
    ### Preprocessing:
    - Pembersihan teks (remove punctuation, lowercasing)
    - Tokenisasi dengan IndoBERT tokenizer
    - Pembatasan panjang teks: 96 token
    - Augmentasi data untuk kelas minoritas
    
    Dataset dikurasi secara manual untuk memastikan kualitas label dan relevansi dengan konteks Indonesia.
    """)

# Examples
with st.expander("📝 Contoh Teks untuk Dicoba"):
    example_texts = [
        "Hari ini rasanya berat, semua tidak berjalan sesuai harapan. Ada perasaan hampa yang sulit dijelaskan.",
        "Sangat kesal dengan situasi yang tidak adil ini! Sudah seharusnya ada perubahan sistem yang lebih baik.",
        "Saya yakin kebenaran akan terbukti, mari kita dukung proses yang adil untuk semua pihak yang terlibat.",
        "Meski sulit sekarang, saya optimis masa depan akan lebih baik. Kita harus tetap semangat dan berusaha.",
        "Mengecewakan melihat bagaimana sistem memperlakukan orang baik. Seharusnya ada reward untuk kejujuran."
    ]
    
    example_labels = [
        "Kesedihan (SADNESS)",
        "Kemarahan (ANGER)",
        "Dukungan (SUPPORT)", 
        "Harapan (HOPE)",
        "Kekecewaan (DISAPPOINTMENT)"
    ]
    
    st.markdown("Pilih contoh teks untuk menguji model:")
    
    for i, (example, label) in enumerate(zip(example_texts, example_labels)):
        if st.button(f"Contoh {i+1}: {label}", key=f"example_{i}"):
            st.session_state.example_text = example
    
    if 'example_text' in st.session_state:
        st.text_area("Contoh teks terpilih:", st.session_state.example_text, height=100, key="example_display")

# Simple footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Dibangun menggunakan <strong>Streamlit</strong> dan <strong>Transformers</strong></p>
    <p>Model dibuat oleh <strong>MonyetttRindam</strong></p>
</div>
""", unsafe_allow_html=True)


