import streamlit as st
import google.generativeai as genai
import pysolr
import pandas as pd
import re
from dateutil import parser as dateparser

# --- Configure Gemini ---
genai.configure(api_key="AIzaSyDq2P1TXEzyBVHSc32FhsTDiwcR-qE25YM")

# --- Solr connection ---
SOLR_URL = "http://localhost:8983/solr/core2"
solr = pysolr.Solr(SOLR_URL, always_commit=True, timeout=10)

# --- Fields to display ---
solr_fields = [
    "id", "title", "content_text",
    "date_of_publish", "author", "brand",
    "type", "filename"
]

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

# --- Helper: parse date robustly ---
def parse_date_to_solr(date_str: str) -> str:
    try:
        parsed_date = dateparser.parse(date_str, dayfirst=True, fuzzy=True)
        return parsed_date.strftime("%Y-%m-%dT00:00:00Z")
    except Exception:
        return None

# --- Build Solr query from user query ---
def build_solr_query(user_query: str) -> str:
    q = []

    # --- Detect brand ---
    brand_match = re.search(r'brand\s+(?:is|=)?\s*"?([\w\s]+)"?', user_query, re.I)
    if brand_match:
        q.append(f'Brand:"{brand_match.group(1)}"')

    # --- Detect author ---
    author_match = re.search(r'author\s+(?:is|=|by)?\s*"?([\w\s]+)"?', user_query, re.I)
    if author_match:
        q.append(f'Author:"{author_match.group(1)}"')

    # --- Detect type ---
    type_match = re.search(r'type\s+(?:is|=)?\s*"?([\w\s]+)"?', user_query, re.I)
    if type_match:
        q.append(f'Type:"{type_match.group(1)}"')

    # --- Detect filename ---
    filename_match = re.search(r'filename\s+(?:is|=)?\s*"?([\w\-.]+)"?', user_query, re.I)
    if filename_match:
        q.append(f'filename:"{filename_match.group(1)}"')

    # --- Detect "before" date ---
    before_match = re.search(r'before\s+([\w\d\s,/-]+)', user_query, re.I)
    if before_match:
        solr_date = parse_date_to_solr(before_match.group(1))
        if solr_date:
            q.append(f'Date_of_publish:[* TO {solr_date}]')

    # --- Detect "after" date ---
    after_match = re.search(r'after\s+([\w\d\s,/-]+)', user_query, re.I)
    if after_match:
        solr_date = parse_date_to_solr(after_match.group(1))
        if solr_date:
            q.append(f'Date_of_publish:[{solr_date} TO NOW]')

    # --- Detect "last X days" ---
    last_days_match = re.search(r'last\s+(\d+)\s+days', user_query, re.I)
    if last_days_match:
        days = last_days_match.group(1)
        q.append(f'Date_of_publish:[NOW-{days}DAYS TO NOW]')

    # --- Detect content phrase search ---
    content_match = re.search(r'content\s+(?:has|contains|with)?\s*"?([\w\s]+)"?', user_query, re.I)
    if content_match:
        q.append(f'content_text:"{content_match.group(1)}"')

    # --- If no explicit filter, search across multiple fields ---
    if not q:
        q.append(f'(title:({user_query}) OR content_text:({user_query}) OR '
                 f'author:({user_query}) OR brand:({user_query}) OR '
                 f'type:({user_query}) OR filename:({user_query}))')

    return " AND ".join(q)

# --- Streamlit UI ---
st.set_page_config(page_title="Hybrid Search (Gemini + Solr)", layout="centered")
st.title("🔎 Hybrid Search with Improved Date Parsing")

user_query = st.text_input("💬 Your Question:", placeholder='e.g. documents of type procedures created by author1 before 21st July 2025')
use_rerank = st.checkbox("Use Gemini Reranking", value=True)

if st.button("Search") and user_query.strip():
    with st.spinner("Searching..."):
        try:
            # Get embedding
            embedding = get_embedding(user_query)
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            # Build query
            solr_query = build_solr_query(user_query)
            st.write(f"**Solr Query Executed:** `{solr_query}`")

            # Hybrid search (embedding + keyword)
            semantic_results = solr.search(solr_query, **{
                "defType": "edismax",
                "qf": "content_text^2 title author brand type filename",
                "knn": f"{{!knn f=content_embedding topK=10}}{embedding_str}",
                "fl": "*,score",
                "rows": 10,
                "mm": "1<50%"
            })

            # Fallback to keyword only
            results = semantic_results if len(semantic_results) > 0 else solr.search(
                solr_query,
                **{
                    "defType": "edismax",
                    "qf": "content_text^2 title author brand type filename",
                    "fl": "*,score",
                    "rows": 10,
                    "mm": "1<50%"
                }
            )

            # Score filtering
            filtered_results = [doc for doc in results if float(doc.get("score", 0)) > 0.6]

            # Gemini Reranking
            if use_rerank and filtered_results:
                filtered_results = rerank_with_gemini(user_query, filtered_results)

            # Display results
            if filtered_results:
                docs = [{field: doc.get(field, "") for field in solr_fields} for doc in filtered_results]
                st.success(f"Found {len(filtered_results)} relevant results")
                st.dataframe(pd.DataFrame(docs), use_container_width=True)
            else:
                st.warning("No relevant results found.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
