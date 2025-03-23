import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from groq import Groq, APIError
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi Groq client
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Gagal menginisialisasi Groq client: {str(e)}")
    st.stop()

# Variabel global untuk model dan tokenizer
tokenizer = None
model = None
context_documents = []

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            logger.info("Memuat model IndoBERT...")
            tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
            model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
            logger.info("Model IndoBERT berhasil dimuat")
        except Exception as e:
            st.error(f"Gagal memuat model: {str(e)}")
            st.stop()

def generate_embeddings(texts, batch_size=16):
    load_model()  # Pastikan model dimuat
    cleaned_texts = [str(text) for text in texts if pd.notna(text) and text != ""]
    if not cleaned_texts:
        raise ValueError("Tidak ada teks valid untuk diproses")
    
    embeddings = []
    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i:i + batch_size]
        logger.info(f"Memproses batch {i//batch_size + 1}")
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def load_context_documents():
    global context_documents
    context_documents.clear()
    context_dir = 'documents'
    if not os.path.exists(context_dir):
        os.makedirs(context_dir)
    for filename in os.listdir(context_dir):
        file_path = os.path.join(context_dir, filename)
        try:
            logger.info(f"Memuat file: {filename}")
            if filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                texts = df.iloc[:, 0].fillna("").astype(str).tolist()
                labels = df.iloc[:, 1].tolist() if df.shape[1] > 1 else [None]*len(texts)
            else:
                continue
            embeddings = generate_embeddings(texts)
            for text, label, embedding in zip(texts, labels, embeddings):
                context_documents.append({'text': text, 'label': label, 'embedding': embedding})
        except Exception as e:
            logger.error(f"Gagal memuat file {filename}: {str(e)}")
            st.warning(f"Gagal memuat file {filename}: {str(e)}")

def find_similar_contexts(input_text, top_n=3):
    input_embedding = generate_embeddings([input_text])
    similarities = []
    for doc in context_documents:
        sim = cosine_similarity(input_embedding, [doc['embedding']])[0][0]
        similarities.append((doc, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in similarities[:top_n]]

def classify_text(text):
    similar_contexts = find_similar_contexts(text)
    context_str = "\n".join(
        [f"Contoh: \"{doc['text']}\" -> {doc['label']}" 
         for doc in similar_contexts if doc['label']]
    )
    prompt = f"""
    Klasifikasikan teks berikut menjadi STRATEGIS, TAKTIKAL, atau OPERASIONAL:
    {context_str}
    
    Teks yang perlu diklasifikasi:
    "{text}"
    
    Jawaban harus dalam format:
    KATEGORI: [STRATEGIS/TAKTIKAL/OPERASIONAL]
    """
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                timeout=30
            )
            return response.choices[0].message.content.strip().split(':')[-1].strip()
        except APIError as e:
            logger.error(f"Percobaan {attempt + 1} gagal: {str(e)}")
            if attempt < 2:
                import time
                time.sleep(2)
            else:
                return f"Error: Gagal mengakses API Groq setelah 3 percobaan"

# UI Streamlit
st.title("Classifier Program")
st.write("Klasifikasi Program Budaya/Kegiatan/Deliverables")

if st.sidebar.button("Muat Ulang Context Documents"):
    with st.spinner("Memuat dokumen context..."):
        load_context_documents()
        st.sidebar.success("Dokumen context dimuat ulang!")

input_text = st.text_area("Masukkan teks untuk diklasifikasi:")
if st.button("Klasifikasi"):
    if input_text:
        with st.spinner("Mengklasifikasi..."):
            result = classify_text(input_text)
            st.info(f"Hasil: {result}")
    else:
        st.warning("Masukkan teks terlebih dahulu!")

# Workaround untuk konflik torch.classes
try:
    import torch._classes
    def safe_getattr(self, name):
        if name == "__path__":
            return None  # Kembalikan None untuk menghindari error
        return torch._C._get_custom_class_python_wrapper(self.name, name)
    torch._classes.__getattr__ = safe_getattr
except ImportError:
    pass  # Import gagal tidak akan menghentikan aplikasi
