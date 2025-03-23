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

# Inisialisasi session state
if 'context_documents' not in st.session_state:
    st.session_state.context_documents = []

# Konfigurasi direktori
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

# Fungsi pembuatan embeddings
def generate_embeddings(texts):
    try:
        inputs = tokenizer(
            texts,
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
                        'label': label,
                        'embedding': emb
                    })
        except Exception as e:
            logging.warning(f"Gagal memproses {file_path}: {str(e)}")
    return documents

# Pembaruan konteks saat startup
st.session_state.context_documents = load_context_documents()

# Fungsi pencarian konteks mirip
def find_similar_contexts(input_text, top_n=3):
    input_emb = generate_embeddings([input_text])
    if input_emb is None:
        return []
    
    similarities = []
    for doc in st.session_state.context_documents:
        sim = cosine_similarity(input_emb, [doc['embedding']])[0][0]
        similarities.append((doc, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in similarities[:top_n]]

# Fungsi klasifikasi
def classify_text(text):
    try:
        similar_contexts = find_similar_contexts(text)
        context_examples = "\n".join(
            [f"- {doc['text']} → {doc['label']}" 
             for doc in similar_contexts if doc['label']]
        ) or "Tidak ada contoh konteks yang tersedia"
        
        prompt = f"""
        Klasifikasikan teks berikut menjadi STRATEGIS, TAKTIKAL, atau OPERASIONAL:
        Contoh Konteks:
        {context_examples}
        
        Teks Target:
        "{text}"
        
        Format jawaban: KATEGORI: [STRATEGIS/TAKTIKAL/OPERASIONAL]
        """
        
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        return response.choices[0].message.content.split(':')[-1].strip()
    
    except Exception as e:
        logging.error(f"Error klasifikasi: {str(e)}")
        return "ERROR"

# Konfigurasi Groq client
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Kesalahan konfigurasi API: {str(e)}")
    st.stop()

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
                result = classify_text(input_text)
                st.success(f"Hasil: {result}")

st.markdown("""
### Catatan Penggunaan:
1. Pastikan konteks sudah diupload di sidebar
2. Format file Excel harus memiliki kolom teks dan label
3. Hasil klasifikasi otomatis disimpan dalam format Excel
""")
