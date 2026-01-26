# 📊 Sentiment Analysis Instagram Comments using Transformer BERT

## 📌 Overview
This project performs **sentiment analysis on Instagram comments** related to the **Tom Lembong case** using a **Transformer-based BERT model**.  
The goal is to classify public sentiment based on real-world social media data using state-of-the-art natural language processing techniques.

## 🚀 Live Demo
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-red?logo=streamlit)](https://tomlembong-euxzfvxsunt7jqrscrkwcb.streamlit.app/)

## 🧠 Model
- Transformer-based **BERT** (Bidirectional Encoder Representations from Transformers)
- Fine-tuned for sentiment classification
- Works on Indonesian / multilingual text input

## 🗂 Dataset
- **Source**: Instagram comments related to the Tom Lembong case  
- **Total Data**: 6,778 samples  
  - **Training Data**: 5,083 comments  
  - **Testing Data**: 1,695 comments  

### 🔄 Preprocessing
- Text cleaning (removing punctuation, hashtags, mentions, URLs)
- Lowercasing
- Tokenization with BERT tokenizer
- Padding & truncation to fixed length

## 📊 Sentiment Classes
- 😃 **Positive**
- 😐 **Neutral**
- 😞 **Negative**

## ⚙️ Tech Stack
- Python
- Hugging Face Transformers
- TensorFlow / PyTorch
- Scikit-learn
- Pandas, NumPy
