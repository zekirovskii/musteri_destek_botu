import streamlit as st
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings # BGE modelleri için bu kütüphane kullanılır
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

import os
import tempfile

# streamlit ile sayfa başlıgı ve icon

st.set_page_config(page_title="Müşteri Destek Botu", page_icon="📃")
st.title("Müşteri Destek Botu (RAG + Memory)")
st.write("Bir pdf yükleyin, içeriğine dair sorular sorun. Türkçe desteklidir.")

# pdf yükleme
upload_file = st.file_uploader("PDF dosyanızı yükleyin.", type = "pdf", key = "pdf_uploader")

# eger kullanıcı yeni bir pdf yuklediyse ve daha önceki yüklenen ile aynı değilse
if upload_file is not None:
    if "last_uploaded_name" not in st.session_state or upload_file.name != st.session_state.last_uploaded_name:
        # kullanıcıya işleniyor bilgisi gönder
        with st.spinner("pdf işleniyor..."):
            # yüklenen pdfi geçici bir dosyaya yazdır
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(upload_file.getvalue())
                tmp_path = tmp.name # geçici dosyanın yolu

            # pdf yükle
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # metinleri parcala yani chunklara böl
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)

            # LaBSE embedding ile metin vektörleştirme
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")

            # faiss ile vektör veritabanı
            vectordb = FAISS.from_documents(docs, embedding)

            # memory ve gemma3:1b yi tanımla
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            llm = ChatOllama(model = "gemma3:1b", temperature = 0.2)

            # rag + memory zinciri
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever = vectordb.as_retriever(search_kwargs = {"k":3}),
                memory = memory
            )

            st.session_state.qa_chain = qa_chain
            st.session_state.chat_history = []
            st.session_state.last_uploaded_name = upload_file.name
        
        st.success("PDF başarıyla işlendi.")
