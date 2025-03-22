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

# Inisialisasi Groq client dengan konfigurasi yang benar
client = Groq(api_key=st.secrets.groq.api_key)

# Load model IndoBERT dengan error handling
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "indolem/indobert-base-uncased",
        force_download=True
    )
    model = AutoModel.from_pretrained(
        "indolem/indobert-base-uncased",
        force_download=True
    )
except Exception as e:
    st.error(f"Gagal memuat model: {str(e)}")
    st.stop()

# Context documents storage
context_documents = []

def generate_embeddings(texts):
    # Tambahkan parameter max_length dan truncation
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def load_context_documents():
    global context_documents
    context_dir = 'documents'
    if not os.path.exists(context_dir):
        os.makedirs(context_dir)
        
    for filename in os.listdir(context_dir):
        file_path = os.path.join(context_dir, filename)
        if filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            texts = df.iloc[:, 0].tolist()
            labels = df.iloc[:, 1].tolist() if df.shape[1] > 1 else [None]*len(texts)
        elif filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            texts = df.iloc[:, 0].tolist()
            labels = df.iloc[:, 1].tolist() if df.shape[1] > 1 else [None]*len(texts)
        elif filename.endswith('.docx'):
            doc = docx.Document(file_path)
            texts = [para.text for para in doc.paragraphs]
            labels = [None]*len(texts)
        else:
            continue
            
        embeddings = generate_embeddings(texts)
        for text, label, embedding in zip(texts, labels, embeddings):
            context_documents.append({
                'text': text,
                'label': label,
                'embedding': embedding
            })

load_context_documents()

def find_similar_contexts(input_text, top_n=3):
    input_embedding = generate_embeddings([input_text])
    similarities = []
    for doc in context_documents:
        sim = cosine_similarity(input_embedding, [doc['embedding']])[0][0]
        similarities.append((doc, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in similarities[:top_n]]

# Streamlit UI
st.title('Classifier Program Budaya/Kegiatan/Deliverables')
st.write('Klasifikasi Program Budaya/Kegiatan/Deliverables menjadi STRATEGIS, TAKTIKAL, atau OPERASIONAL')

# Sidebar untuk upload data context
st.sidebar.header('Upload Data Context')
context_file = st.sidebar.file_uploader(
    "Upload file Excel/CSV/DOCX untuk context",
    type=['xlsx', 'csv', 'docx']
)

if context_file is not None:
    try:
        save_path = os.path.join('documents', context_file.name)
        with open(save_path, 'wb') as f:
            f.write(context_file.getvalue())
        st.sidebar.success("File context berhasil disimpan!")
        load_context_documents()  # Reload context
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# Main classification area
st.header('Klasifikasi Data Baru')

tab1, tab2 = st.tabs(["Upload File", "Input Manual"])

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
    {text}
    
    Jawaban harus dalam format:
    KATEGORI: [STRATEGIS/TAKTIKAL/OPERASIONAL]
    """
    
    response = client.chat.completions.create(
        model="llama-3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    
    return response.choices[0].message.content.strip().split(':')[-1].strip()

with tab1:
    pred_file = st.file_uploader(
        "Upload file Excel untuk klasifikasi",
        type=['xlsx']
    )
    
    if pred_file is not None:
        try:
            pred_df = pd.read_excel(pred_file)
            st.success("File berhasil diupload!")
            pred_column = st.selectbox(
                'Pilih kolom teks yang akan diklasifikasi:',
                pred_df.columns
            )
            
            if st.button('Klasifikasi File'):
                with st.spinner('Mengklasifikasi data...'):
                    results = []
                    for text in pred_df[pred_column]:
                        results.append(classify_text(text))
                    
                    pred_df['Hasil_Klasifikasi'] = results
                    st.write('Hasil Klasifikasi:')
                    st.dataframe(pred_df)
                    
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        pred_df.to_excel(writer, index=False)
                    st.download_button(
                        label="Download hasil klasifikasi (Excel)",
                        data=buffer.getvalue(),
                        file_name="hasil_klasifikasi.xlsx",
                        mime="application/vnd.ms-excel"
                    )
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab2:
    input_text = st.text_area(
        "Tuliskan Program Budaya/Kegiatan/Deliverables Anda di bawah ini:"
    )
    
    if st.button('Klasifikasi Teks'):
        if not input_text:
            st.warning("Mohon masukkan teks terlebih dahulu!")
        else:
            with st.spinner('Mengklasifikasi teks...'):
                result = classify_text(input_text)
                st.write('Hasil Klasifikasi:')
                st.info(f"Teks termasuk kategori: {result}")

st.markdown("""
### Petunjuk Penggunaan:
1. Pastikan sudah mengunggah file context di sidebar
2. Untuk input file Excel, pastikan format kolom sesuai
3. Hasil klasifikasi bisa didownload dalam format Excel
""")
