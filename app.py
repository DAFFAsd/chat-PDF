import streamlit as st
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile

# Add logo at the top-left
st.image(
    "https://beta.sisva.id/_next/image?url=https%3A%2F%2Fwww.sisva.id%2Fimages%2FSisva-LogoType-Black.png&w=384&q=75",
    width=150  # Adjust the width as needed
)

# Get API keys from secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Set Google API key as environment variable
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize ChatGroq with the secret API key
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="deepseek-r1-distill-llama-70b")

# Konfigurasi sidebar
with st.sidebar:
    # Bagian yang dapat diperluas untuk informasi aplikasi
    with st.expander("🔍 Tentang Aplikasi", expanded=True):
        st.write(
            "Selamat datang di **SISVA Chat with PDF**! Alat ini memungkinkan Anda berinteraksi dengan dokumen PDF dengan mudah. "
            "Unggah dokumen Anda dan ajukan pertanyaan tentang isinya secara langsung."
        )

    # Bagian yang dapat diperluas dengan petunjuk penggunaan
    with st.expander("📝 Panduan", expanded=False):
        st.write(
            "Ikuti langkah-langkah berikut untuk menggunakan **Chat with PDF**:\n\n"
            "1. **📄 Unggah PDF**: Pilih dan unggah PDF yang ingin Anda interaksikan.\n"
            "2. **🔍 Proses Dokumen**: Klik 'Proses Dokumen' untuk menganalisis PDF.\n"
            "3. **💬 Mulai Chat**: Ajukan pertanyaan di kotak chat untuk menerima respons.\n"
            "4. **📑 Lihat Konteks**: Bagian relevan dari dokumen akan ditampilkan dalam chat."
        )

    # Tentukan template prompt chat
    prompt = ChatPromptTemplate.from_template(
        """
        Anda adalah asisten pendidikan dari badan bernama SISVA.ID yang menggunakan bahasa utama yaitu indonesia dengan sopan dan informatif. 
        Anda mampu membantu dalam berbagai hal yang berkaitan dengan pendidikan, seperti materi pelajaran, tugas, dan lainnya. 
        Terkhususnya dari konteks file .pdf yang diberikan ke Anda.
        
        Jawablah pertanyaan berdasarkan konteks yang diberikan.
        Mohon berikan respons yang paling akurat berdasarkan pertanyaan.
        
        <context>
        {context}
        </context>
        
        Pertanyaan: {input}
        """
    )

    # Pengunggah file untuk beberapa PDF
    uploaded_files = st.file_uploader(
        "Unggah PDF", type="pdf", accept_multiple_files=True
    )

    # Proses PDF yang diunggah ketika tombol diklik
    if uploaded_files:
        if st.button("Proses Dokumen"):
            with st.spinner("Memproses dokumen... Mohon tunggu."):

                def vector_embedding(uploaded_files):
                    if "vectors" not in st.session_state:
                        # Inisialisasi embedding jika belum dilakukan
                        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/embedding-001"
                        )
                        all_docs = []

                        # Proses setiap file yang diunggah
                        for uploaded_file in uploaded_files:
                            # Simpan file yang diunggah sementara
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".pdf"
                            ) as temp_file:
                                temp_file.write(uploaded_file.read())
                                temp_file_path = temp_file.name

                            # Muat dokumen PDF
                            loader = PyPDFLoader(temp_file_path)
                            docs = loader.load()  # Muat konten dokumen

                            # Hapus file sementara
                            os.remove(temp_file_path)

                            # Tambahkan dokumen yang dimuat ke daftar
                            all_docs.extend(docs)

                        # Bagi dokumen menjadi bagian-bagian yang dapat dikelola
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, chunk_overlap=200
                        )
                        final_documents = text_splitter.split_documents(all_docs)

                        # Buat penyimpanan vektor dengan FAISS
                        st.session_state.vectors = FAISS.from_documents(
                            final_documents, st.session_state.embeddings
                        )

                vector_embedding(uploaded_files)
                st.sidebar.write("Dokumen berhasil diproses :partying_face:")

# Area utama untuk antarmuka chat
st.title("Chat with PDF :speech_balloon:")

# Inisialisasi state sesi untuk pesan chat jika belum dilakukan
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field untuk pertanyaan pengguna
if human_input := st.chat_input("Tanyakan sesuatu tentang dokumen"):
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        # Buat dan konfigurasi rantai dokumen dan pengambil
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Dapatkan respons dari asisten
        response = retrieval_chain.invoke({"input": human_input})
        assistant_response = response["answer"]

        # Tambahkan dan tampilkan respons asisten
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Tampilkan informasi pendukung dari dokumen
        with st.expander("Informasi Pendukung"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        # Minta pengguna untuk mengunggah dan memproses dokumen jika tidak ada vektor yang tersedia
        assistant_response = (
            "Silakan unggah dan proses dokumen sebelum mengajukan pertanyaan."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/api/query")
async def query(request: Request, query: Query):
    human_input = query.question
    if "vectors" in st.session_state and st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": human_input})
        assistant_response = response["answer"]
        return JSONResponse(content={"response": assistant_response})
    else:
        return JSONResponse(content={"response": "Silakan unggah dan proses dokumen sebelum mengajukan pertanyaan."})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
