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