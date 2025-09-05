import streamlit as st
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import os
from huggingface_hub import hf_hub_download
from tensorflow.keras.utils import get_custom_objects
from transformers import TFAutoModelForSequenceClassification, BertConfig

# Konfigurasi halaman
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🔍",
    layout="wide"
)

# Mapping sentimen
SENTIMENT_MAPPING = {
    0: "SADNESS", 
    1: "ANGER", 
    2: "SUPPORT", 
    3: "HOPE", 
    4: "DISAPPOINTMENT"
}

# Konstanta
SEQ_LEN = 96

# Fungsi preprocessing
URL_RE = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
MENTION_RE = re.compile(r'@\w+')
REPEAT_PAT = re.compile(r'(.)\1{2,}')

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(' ', text)
    text = MENTION_RE.sub(' ', text)
    text = re.sub(r'[^0-9a-zA-Z_\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_repeated_chars(text: str) -> str:
    return REPEAT_PAT.sub(r'\1\1', str(text))

def normalize_slang(text, slang_map):
    if not isinstance(text, str):
        return ""
    for slang, norm in slang_map.items():
        text = re.sub(re.escape(slang), f" {norm} ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Slang mapping
slang_map = {
    '❤️': 'cinta', '😍': 'gembira', '😭': 'sedih', '😢': 'sedih', 
    '😡': 'marah', '😠': 'marah', '😂': 'tertawa', '🤣': 'tertawa', 
    '😅': 'gembira', '😊': 'gembira', '👍': 'bagus', '👎': 'buruk', 
    '🤔': 'berpikir', '😱': 'kaget', '😤': 'kesal', '😞': 'sedih', 
    '🤯': 'kaget', '🥰': 'gembira'
}

def full_preprocess(text):
    """Fungsi preprocessing lengkap"""
    text = preprocess(text)
    text = normalize_repeated_chars(text)
    text = normalize_slang(text, slang_map)
    return text

@st.cache_resource
def load_model_and_tokenizer():
    """Load model dan tokenizer dari Hugging Face"""
    try:
        # Download model dari Hugging Face
        model_path = hf_hub_download(
            repo_id="MonyetttRindam/tomlembong",
            filename="sentimentanalyisisl.h5"
        )
        
        # Gunakan TFAutoModelForSequenceClassification untuk klasifikasi
        with tf.keras.utils.custom_object_scope({'TFBertModel': TFAutoModelForSequenceClassification}):
            model = load_model(model_path)
        
        # Untuk tokenizer, Anda perlu mengupload file tokenizer juga ke HF
        # Atau buat tokenizer baru (tidak ideal untuk produksi)
        try:
            tokenizer_path = hf_hub_download(
                repo_id="MonyetttRindam/tomlembong",
                filename="tokenizer.pickle"
            )
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
        except:
            # Fallback: buat tokenizer baru (tidak ideal)
            st.warning("Tokenizer tidak ditemukan. Menggunakan tokenizer default.")
            tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
            
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_sentiment(text, model, tokenizer):
    """Prediksi sentimen dari teks"""
    try:
        # Preprocess text
        processed_text = full_preprocess(text)
        
        # Tokenisasi
        sequences = tokenizer.texts_to_sequences([processed_text])
        
        # Padding
        padded_sequences = pad_sequences(sequences, maxlen=SEQ_LEN, padding='post')
        
        # Prediksi
        prediction = model.predict(padded_sequences)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return SENTIMENT_MAPPING[predicted_class], confidence, prediction[0]
        
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return None, None, None

def main():
    st.title("🔍 Sentiment Analysis App")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Gagal memuat model. Silakan periksa konfigurasi Hugging Face.")
        st.stop()
    
    st.success("Model berhasil dimuat!")
    
    # Input text
    st.subheader("📝 Input Teks")
    user_input = st.text_area(
        "Masukkan teks yang ingin dianalisis:",
        placeholder="Contoh: Saya sangat senang dengan hasil ini!",
        height=100
    )
    
    # Tombol prediksi
    if st.button("🔮 Analisis Sentimen", type="primary"):
        if user_input.strip():
            with st.spinner("Menganalisis sentimen..."):
                sentiment, confidence, all_predictions = predict_sentiment(
                    user_input, model, tokenizer
                )
            
            if sentiment:
                # Hasil prediksi
                st.subheader("📊 Hasil Analisis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Sentimen Terdeteksi", sentiment)
                    st.metric("Confidence Score", f"{confidence:.2%}")
                
                with col2:
                    # Progress bar untuk confidence
                    st.write("**Tingkat Kepercayaan:**")
                    st.progress(confidence)
                
                # Detail semua prediksi
                with st.expander("📈 Detail Semua Prediksi"):
                    for i, (label, score) in enumerate(zip(SENTIMENT_MAPPING.values(), all_predictions)):
                        st.write(f"**{label}**: {score:.4f} ({score:.2%})")
                        st.progress(score)
                
                # Teks yang diproses
                with st.expander("🔧 Teks Setelah Preprocessing"):
                    processed = full_preprocess(user_input)
                    st.write(f"Original: `{user_input}`")
                    st.write(f"Processed: `{processed}`")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu!")
    
    # Informasi aplikasi
    st.markdown("---")
    with st.expander("ℹ️ Informasi Aplikasi"):
        st.write("""
        **Model Information:**
        - Model: sentimentanalyisisfinal.h5
        - Repository: MonyetttRindam/tomlembong
        - Tokenizer: IndoBERT (indobenchmark/indobert-base-p2)
        - Architecture: BERT + Dense layers
        - Inputs: input_ids + attention_mask
        - Sequence Length: 96
        
        **Sentiment Categories:**
        - SADNESS (Kesedihan)
        - ANGER (Kemarahan) 
        - SUPPORT (Dukungan)
        - HOPE (Harapan)
        - DISAPPOINTMENT (Kekecewaan)
        
        **Preprocessing Steps:**
        1. Text cleaning (URL, mentions, special chars removal)
        2. Normalisasi karakter berulang
        3. Konversi emoji dan slang ke teks
        """)

if __name__ == "__main__":
    main()
