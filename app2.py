import streamlit as st
import google.generativeai as genai
import pysolr
import pandas as pd

# Configure Gemini
genai.configure(api_key="AIzaSyDq2P1TXEzyBVHSc32FhsTDiwcR-qE25YM")

# Solr connection
SOLR_URL = "http://localhost:8983/solr/file"
solr = pysolr.Solr(SOLR_URL, always_commit=True, timeout=10)

# Fields to display
solr_fields = ["id", "title", "content_text"]

# --- Embedding ---
def get_embedding(text: str) -> list:
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="semantic_similarity"
    )
    return response["embedding"]

# --- Rerank with Gemini ---
def rerank_with_gemini(query, docs):
    if not docs:
        return []
    items = "\n".join([f"Document {i+1}: {doc.get('content_text','')[:300]}" 
                       for i, doc in enumerate(docs)])
    prompt = f"""
    You are an expert search assistant.
    Rerank these documents for relevance to the query: "{query}"
    Documents:
    {items}
    Return the best document order as a JSON list of integers (indexes starting from 1).
    """
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    try:
        order = eval(response.text.strip())
        return [docs[i-1] for i in order if 1 <= i <= len(docs)]
    except:
        return docs

# --- Streamlit UI ---
st.set_page_config(page_title="Hybrid Search (Gemini + Solr)", layout="centered")
st.title("🔎 Hybrid Search with Score Filtering")

user_query = st.text_input("💬 Your Question:", placeholder="e.g. How to fight bacterial infection?")
use_rerank = st.checkbox("Use Gemini Reranking", value=True)

if st.button("Search") and user_query.strip():
    with st.spinner("Searching..."):
        try:
            # Get embedding
            embedding = get_embedding(user_query)
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            # Semantic search (vector)
            semantic_results = solr.search("*:*", **{
                "knn": f"{{!knn f=content_embedding topK=10}}{embedding_str}",
                "fl": "*,score",
                "rows": 10
            })

            # If no semantic match, fallback to keyword
            if len(semantic_results) == 0:
                results = solr.search(user_query, **{
                    "defType": "edismax",
                    "qf": "content_text title",
                    "fl": "*,score",
                    "rows": 10
                })
            else:
                results = semantic_results

            # Score filtering (remove irrelevant docs)
            filtered_results = [doc for doc in results if float(doc.get("score", 0)) > 0.2]

            # Gemini Reranking
            if use_rerank and filtered_results:
                filtered_results = rerank_with_gemini(user_query, filtered_results)

            # Display
            if filtered_results:
                docs = [{field: doc.get(field, "") for field in solr_fields} for doc in filtered_results]
                st.success(f"Found {len(filtered_results)} relevant results")
                st.dataframe(pd.DataFrame(docs), use_container_width=True)
            else:
                st.warning("No relevant results found.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
