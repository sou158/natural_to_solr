import streamlit as st
import google.generativeai as genai
import pysolr
import pandas as pd
from typing import List

# ========== 1. Configure Gemini ==========
genai.configure(api_key="AIzaSyDq2P1TXEzyBVHSc32FhsTDiwcR-qE25YM")  # <-- Replace with your key

# ========== 2. Solr Setup ==========
SOLR_URL = "http://localhost:8983/solr/file"  # <-- Replace with your Solr core URL
solr = pysolr.Solr(SOLR_URL, always_commit=True, timeout=10)

# ========== 3. Solr Fields to Show ==========
solr_fields = ["id", "title", "content_text"]

# ========== 4. Gemini Embedding ==========
def get_embedding(text: str) -> list:
    
    response = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    return response["embedding"]

# ========== 5. Merge Results with Scores ==========
def merge_results(semantic_results, keyword_results):
    combined = {}
    for doc in semantic_results:
        combined[doc["id"]] = {"doc": doc, "score": 2 * float(doc.get("score", 1))}
    for doc in keyword_results:
        if doc["id"] in combined:
            combined[doc["id"]]["score"] += float(doc.get("score", 1))
        else:
            combined[doc["id"]] = {"doc": doc, "score": float(doc.get("score", 1))}
    sorted_docs = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return [x["doc"] for x in sorted_docs]

# ========== 6. Gemini Reranking ==========
def rerank_with_gemini(query: str, docs: List[dict]) -> List[dict]:
    if not docs:
        return []
    items = "\n".join(
        [f"Document {i+1}: {doc.get('content_text', '')[:300]}" for i, doc in enumerate(docs)]
    )
    prompt = f"""
    You are an expert search assistant.
    Rerank these documents for relevance to the query: "{query}"
    Documents:
    {items}
    Return the best document order as a JSON list of integers (indexes starting from 1).
    """
    model = genai.GenerativeModel("models/text-bison-001")
    response = model.generate_content(prompt)
    try:
        order = eval(response.text.strip())
        return [docs[i - 1] for i in order if 1 <= i <= len(docs)]
    except Exception:
        return docs  # fallback if Gemini output isn't clean

# ========== 7. Streamlit UI ==========
st.set_page_config(page_title="Hybrid Search (Gemini + Solr)", layout="centered")
st.title("🔎 Hybrid Search: Semantic + Keyword (with Optional Reranking)")

user_query = st.text_input("💬 Your Question:", placeholder="e.g. Explain symptoms of depression or anxiety")
use_rerank = st.checkbox("Use Gemini Reranking", value=True)

if st.button("Search") and user_query.strip():
    with st.spinner("Searching..."):
        try:
            # Step 1: Get Embedding
            embedding = get_embedding(user_query)
            embedding_str = ",".join(map(str, embedding))

            # Step 2: Semantic Search (Solr KNN)
            semantic_results = solr.search(
                q=f"{{!knn f=content_embedding topK=20}}{embedding_str}",
                **{"fl": "*", "rows": 10}
            )

            # Step 3: Keyword Search
            keyword_results = solr.search(
                q=user_query,
                **{"defType": "edismax", "qf": "content_text title", "fl": "*", "rows": 10}
            )

            # Step 4: Merge Results
            results = merge_results(semantic_results, keyword_results)

            # Step 5 (Optional): Rerank
            if use_rerank and results:
                results = rerank_with_gemini(user_query, results)

            # Step 6: Display
            if results:
                docs = [{field: doc.get(field, "") for field in solr_fields} for doc in results]
                st.success(f"Found {len(results)} results")
                st.dataframe(pd.DataFrame(docs), use_container_width=True)
            else:
                st.warning("No results found.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
