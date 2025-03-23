import streamlit as st
import pandas as pd
import numpy as np
import os
from groq import Groq
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import docx
from pathlib import Path
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Inisialisasi direktori
DATA_DIR = Path('documents')
DATA_DIR.mkdir(exist_ok=True)

# Inisialisasi model dan tokenizer
@st.cache_resource(show_spinner="Memuat model IndoBERT...")
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
        model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
        return tokenizer, model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.stop()

tokenizer, model = load_model()

# Konfigurasi Groq client
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Kesalahan konfigurasi API: {str(e)}")
    st.stop()

# Session state untuk konteks
if 'context_documents' not in st.session_state:
    st.session_state.context_documents = []

# Fungsi pembuatan embeddings
def generate_embeddings(texts):
    try:
        cleaned_texts = [str(text).strip() for text in texts if pd.notna(text) and text != ""]
        if not cleaned_texts:
            return None
        
        inputs = tokenizer(
            cleaned_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    except Exception as e:
        logging.error(f"Error generate embeddings: {str(e)}")
        return None

# Fungsi pemuatan dokumen konteks
def load_context_documents():
    documents = []
    for file_path in DATA_DIR.glob('*'):
        try:
            if file_path.suffix in ['.xlsx', '.csv']:
                df = pd.read_excel(file_path) if file_path.suffix == '.xlsx' else pd.read_csv(file_path)
                texts = df.iloc[:, 0].fillna("").astype(str).tolist()
                labels = df.iloc[:, 1].tolist() if df.shape[1] > 1 else [None]*len(texts)
            elif file_path.suffix == '.docx':
                doc = docx.Document(file_path)
                texts = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
                labels = [None]*len(texts)
            else:
                continue

            embeddings = generate_embeddings(texts)
            if embeddings is not None:
                for text, label, emb in zip(texts, labels, embeddings):
                    documents.append({
                        'text': text,
                        'label': label.upper() if label else None,
                        'embedding': emb
                    })
        except Exception as e:
            logging.warning(f"Gagal memproses {file_path}: {str(e)}")
    return documents

# Pembaruan konteks
st.session_state.context_documents = load_context_documents()

# Fungsi pencarian konteks mirip dengan threshold
def find_similar_contexts(input_text, top_n=3, threshold=0.65):
    input_emb = generate_embeddings([input_text])
    if input_emb is None:
        return []
    
    similarities = []
    for doc in st.session_state.context_documents:
        if doc['label'] is None:
            continue
        sim = cosine_similarity(input_emb, [doc['embedding']])[0][0]
        if sim >= threshold:
            similarities.append((doc, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in similarities[:top_n]]

# Validasi kategori
ALLOWED_CATEGORIES = {"STRATEGIS", "TAKTIKAL", "OPERASIONAL"}

def validate_category(category):
    upper_cat = category.upper()
    return upper_cat if upper_cat in ALLOWED_CATEGORIES else "TIDAK JELAS"

# Fungsi klasifikasi dengan prompt yang diperbaiki
def classify_text(text):
    try:
        similar_contexts = find_similar_contexts(text)
        context_examples = "\n".join(
            [f"Contoh {i+1}: \"{doc['text']}\" → {doc['label']}" 
             for i, doc in enumerate(similar_contexts) if doc['label']]
        ) or "Tidak ada contoh yang relevan"

        prompt = f"""
        KLASIFIKASI PROGRAM BUDAYA/KEGIATAN
        INSTRUKSI:
        1. Gunakan CONTOH KONTEKS berikut sebagai referensi
        2. Klasifikasikan ke dalam: STRATEGIS, TAKTIKAL, atau OPERASIONAL
        3. Jawaban harus dalam format: KATEGORI: [STRATEGIS/TAKTIKAL/OPERASIONAL]

        CONTOH KONTEKS:
        {context_examples}

        TEKS TARGET:
        "{text}"
        """
        
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1
        )
        result = response.choices[0].message.content.strip()
        category = result.split(':')[-1].strip() if 'KATEGORI:' in result else "TIDAK JELAS"
        return validate_category(category)
    
    except Exception as e:
        logging.error(f"Error klasifikasi: {str(e)}")
        return "ERROR"

# ======== Tampilan Streamlit ========
st.title('Klasifikasi Program Budaya/Kegiatan/Deliverables')
st.write('Klasifikasi ke STRATEGIS, TAKTIKAL, atau OPERASIONAL')

# Sidebar untuk upload konteks
with st.sidebar:
    st.header('Pengaturan Konteks')
    context_file = st.file_uploader(
        "Upload konteks (Excel/CSV/DOCX)",
        type=['xlsx', 'csv', 'docx'],
        key='context_uploader'
    )
    
    if context_file:
        try:
            file_path = DATA_DIR / context_file.name
            with open(file_path, 'wb') as f:
                f.write(context_file.getvalue())
            st.success("Konteks berhasil diperbarui!")
            st.session_state.context_documents = load_context_documents()
        except Exception as e:
            st.error(f"Gagal menyimpan konteks: {str(e)}")

# Tab utama
tab1, tab2 = st.tabs(["Upload File", "Input Manual"])

with tab1:
    pred_file = st.file_uploader(
        "Upload file Excel untuk klasifikasi",
        type=['xlsx'],
        key='pred_uploader'
    )
    
    if pred_file:
        try:
            pred_df = pd.read_excel(pred_file)
            column_options = pred_df.select_dtypes(include=['object']).columns.tolist()
            pred_column = st.selectbox('Pilih kolom teks:', column_options)
            
            if st.button('Mulai Klasifikasi'):
                with st.spinner('Memproses klasifikasi...'):
                    results = []
                    for text in pred_df[pred_column].fillna("").astype(str):
                        results.append(classify_text(text))
                    
                    pred_df['Hasil_Klasifikasi'] = results
                    st.dataframe(pred_df)
                    
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer) as writer:
                        pred_df.to_excel(writer, index=False)
                    st.download_button(
                        label="Download Hasil",
                        data=buffer.getvalue(),
                        file_name="hasil_klasifikasi.xlsx",
                        mime="application/vnd.ms-excel"
                    )
        except Exception as e:
            st.error(f"Error pemrosesan file: {str(e)}")

with tab2:
    input_text = st.text_area("Masukkan teks untuk klasifikasi:")
    if st.button('Klasifikasi'):
        if not input_text.strip():
            st.warning("Teks tidak boleh kosong!")
        else:
            with st.spinner('Memproses...'):
                # Tampilkan contoh yang digunakan
                similar_contexts = find_similar_contexts(input_text)
                st.write("Contoh Referensi yang Digunakan:")
                st.table(pd.DataFrame([{
                    'Teks': doc['text'],
                    'Kategori': doc['label']
                } for doc in similar_contexts]))
                
                result = classify_text(input_text)
                st.write('Hasil Klasifikasi:')
                st.info(f"Kategori: {result}")

st.markdown("""
### Catatan Penting:
1. Pastikan konteks memiliki minimal 10 contoh untuk setiap kategori
2. Format file Excel konteks harus: Kolom 1 = Teks, Kolom 2 = Kategori
3. Contoh kategori yang valid: STRATEGIS, TAKTIKAL, OPERASIONAL
""")
