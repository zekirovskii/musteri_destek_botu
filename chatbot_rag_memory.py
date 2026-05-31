"""
Problem Tanımı: Akıllı müsteri destek sistemi: sık sorulan sorulara yanıt verecek, belgeye dayalı yanıt sistemi
    - Müsteriler sık sık benzer sorular sorarlar:
        - sifremi unuttum,
        - faturamı nereden alabilirim
        - iade süresi kac gün
    - Çözüm: 
        - .pdf (db, text, json ...) dosyası formatında sıkça sorulan soruları vektör veritabanına dönüştürürüz
        - kullanıcıdan gelen sorular veritabanında sorgulanır gemme llm türkçe cevaplar üretir

Kullanılan Teknolojiler:
    - langchain: rag mimari
    - faiss: embeddingleri saklamak için hızlı bir vektör db
    - ollama: gemma3:4b ya da 1b soru cevap llm için
    - streamlit: web arayüzü

Veri seti -> gemini
    - soru: Yurt dısı satıslarınız bulunuyor mu?
    - cevap: Henüz bulunmuyor.

Plan:
    - SSS içeren bir pdf olustur
    - kullanıcı bu dosyayı arayüzden yükleyecek
    - pdf metni chunklara ayrılır, embeddingler cıkarılır
    - kullanıcı soru sordugu zaman vektör dbden benzer içerikler getirilir, gemma ile cevap olusturulur
    - memory ile konusma gecmisi saklanır ve sonraki yanıtlara bağlam olusur

pip install langchain langchain-community sentence-transformers faiss-cpu pypdf ollama streamlit

"""
import os
# Modern yapı
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferMemory

# embedding modelini baslat
embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/LaBSE"
)

# daha önceden olusturulmus vektor db yükle
vectordb = FAISS.load_local(
    "faq_vectorstore",
    embedding,
    allow_dangerous_deserialization = True
)

# konusma gecmisi için memory olustur
memory = ConversationBufferMemory(
    memory_key = "chat_history",
    return_messages = True
)

# llm tanımla
llm = ChatOllama( 
    model = "gemma3:1b",
    temperature = 0.2
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = vectordb.as_retriever(search_kwargs = {"k":3}),
    memory = memory,
    verbose = True
)

# test
print( "Müşteri Destek Botuna Hoş Geldiniz")
while True:
    user_input = input("Siz: ")
    if user_input.lower() == "çık":
        break
    response = qa_chain.run(user_input)
    print(f"Müşteri Destek Botu: {response}")
