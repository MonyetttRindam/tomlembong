import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np

# Cache model loading untuk performa yang lebih baik
@st.cache_resource
def load_model_and_tokenizer():
    """Load model dan tokenizer dengan caching"""
    try:
        # Memuat model dari TensorFlow ke PyTorch
        model = TFAutoModelForSequenceClassification.from_pretrained(
            "MonyetttRindam/emotion_classification_model")
        tokenizer = AutoTokenizer.from_pretrained("MonyetttRindam/emotion_classification_model")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Fungsi untuk memprediksi emosi
def predict_emotion(text, model, tokenizer):
    """Prediksi emosi dari teks input"""
    try:
        if not text or len(text.strip()) == 0:
            return None, None
        
        # Tokenisasi
        inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=96)
        
        # Prediksi dengan TensorFlow
        outputs = model(**inputs)
        predictions = outputs.logits
        
        # Softmax untuk probabilitas
        probabilities = tf.nn.softmax(predictions, axis=-1)
        
        # Ambil predicted class
        predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]
        
        # Label emosi
        emotions = ["SADNESS", "ANGER", "SUPPORT", "HOPE", "DISAPPOINTMENT"]
        
        return emotions[predicted_class], probabilities[0].numpy()
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Streamlit UI
st.title("🎭 Emotion Classification with Fine-Tuned BERT")
st.write("This app uses a fine-tuned BERT model to classify the emotion of a given text.")
st.write("**Supported emotions:** SADNESS, ANGER, SUPPORT, HOPE, DISAPPOINTMENT")

# Load model dan tokenizer
with st.spinner("Loading model..."):
    model, tokenizer = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.error("Failed to load model. Please check your internet connection and try again.")
    st.stop()

st.success("Model loaded successfully!")

# Input box untuk pengguna memasukkan teks
st.subheader("Enter Text for Emotion Classification")
user_input = st.text_area(
    "Enter Text:", 
    placeholder="Type your text here...",
    height=100
)

# Button untuk prediksi
if st.button("Classify Emotion", type="primary"):
    if user_input and user_input.strip():
        with st.spinner("Analyzing emotion..."):
            prediction, probabilities = predict_emotion(user_input, model, tokenizer)
            
            if prediction is not None:
                st.subheader("Results:")
                
                # Tampilkan prediksi utama
                st.success(f"**Predicted Emotion: {prediction}**")
                
                # Tampilkan confidence scores
                st.subheader("Confidence Scores:")
                emotions = ["SADNESS", "ANGER", "SUPPORT", "HOPE", "DISAPPOINTMENT"]
                
                # Buat dataframe untuk visualisasi
                import pandas as pd
                scores_df = pd.DataFrame({
                    'Emotion': emotions,
                    'Confidence': probabilities
                })
                scores_df = scores_df.sort_values('Confidence', ascending=False)
                
                # Bar chart
                st.bar_chart(data=scores_df.set_index('Emotion')['Confidence'])
                
                # Tabel scores
                for emotion, score in zip(emotions, probabilities):
                    st.write(f"**{emotion}**: {score:.4f} ({score*100:.2f}%)")
                    
            else:
                st.error("Failed to predict emotion. Please try again.")
    else:
        st.warning("Please enter some text to classify.")

# Tambahkan informasi tambahan
with st.expander("ℹ️ About this model"):
    st.write("""    
    This emotion classification model is based on BERT and fine-tuned to classify text into 5 emotional categories:
    - **SADNESS**: Expressions of sorrow, grief, or melancholy
    - **ANGER**: Expressions of rage, frustration, or irritation  
    - **SUPPORT**: Expressions of encouragement, help, or solidarity
    - **HOPE**: Expressions of optimism, expectation, or aspiration
    - **DISAPPOINTMENT**: Expressions of dissatisfaction or unmet expectations
    
    The model uses a maximum sequence length of 96 tokens.
    """)

# Contoh teks untuk testing
with st.expander("📝 Try these examples"):
    example_texts = [
        "I'm feeling so down today, everything seems to go wrong.",
        "This is absolutely frustrating! I can't stand this anymore!",
        "You can do it! I believe in you and I'm here to help.",
        "I'm excited about tomorrow, things will get better for sure!",
        "I expected so much more from this, what a letdown."
    ]
    
    for i, example in enumerate(example_texts, 1):
        if st.button(f"Example {i}", key=f"example_{i}"):
            # Set example text ke text area
            st.session_state.example_text = example

    # Display selected example
    if 'example_text' in st.session_state:
        st.text_area("Selected example:", st.session_state.example_text, key="example_display")

        link = 'https://tomlembong-ekl9h5mxnpfbkal8dq4p3e.streamlit.app/'
