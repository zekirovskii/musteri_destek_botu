"""
- kütüphaneleri içeri aktar
- sss pdf olustur ve yükle
- chunkları olustur
- embedding uygula
- vektor db olustur
- vector dbyi kaydet
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# SSS dosyasını yükle
loader = PyPDFLoader("musteri_destek.pdf")
documents = loader.load() # langchain document objesi olustur
#print(documents)

# metinleri parcalama
# splitter: metni anlamlı parcalara ayırma

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, # her parca max 500 karakter
    chunk_overlap = 50 # her parca bir öncekinden 50 karakter alabilir
)

# chunkları olustur
docs = text_splitter.split_documents(documents)

# türkce icin labse embedding yöntemi
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/LaBSE"
)

# parcalara ayrılmıs metni embedding ile vector haline getir ve faiss de depola
vector_db = FAISS.from_documents(docs, embedding)

# olusturulan vektor veri tabanını yerel diske kaydet
vector_db.save_local("faq_vectorstore")

print("Embedding ve vektör veritabanı başarılı bir şekilde oluşturuldu.")
