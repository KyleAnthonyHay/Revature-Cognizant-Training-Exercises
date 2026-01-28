import streamlit as st
import chromadb

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./vector_db")
    return client.get_or_create_collection("synthetic_clothes_data")

collection = get_collection()

st.title("Product Search")
query = st.text_input("Search products")

if query:
    results = collection.query(query_texts=[query], n_results=5)
    
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        st.write(f"**{meta.get('product_name')}** - {meta.get('brand')}")
        st.write(f"${meta.get('price_usd', 0):.2f} | ⭐ {meta.get('avg_rating', 0)}/5")
        st.write(f"Match: {(1-dist)*100:.1f}%")
        st.write(doc)
        st.divider()