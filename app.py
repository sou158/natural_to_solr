import streamlit as st
import google.generativeai as genai
import pysolr
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from prompt_template import get_few_shot_prompt
from langchain.llms.base import LLM
from typing import Optional, List
import re

# ========== 1. Configure Gemini ========== #
genai.configure(api_key="AIzaSyDq2P1TXEzyBVHSc32FhsTDiwcR-qE25YM")  # Replace with your Gemini API key

# ========== 2. Configure Solr ========== #
solr = pysolr.Solr("http://localhost:8983/solr/core4", always_commit=True, timeout=10)

# ========== 3. Solr Fields ========== #
solr_fields = ["id", "title", "content_text", "author", "brand", "type", "date_of_publish"]

# ========== 4. Synonym Map & Expansion ========== #
SYNONYM_MAP = {
    "j&j": "johnson&johnson",
    "sop": "procedural",
}

def expand_synonyms(query: str) -> str:
    words = query.lower().split()
    for i, word in enumerate(words):
        if word in SYNONYM_MAP:
            words[i] = SYNONYM_MAP[word]
    return " ".join(words)

# ========== 5. Gemini LLM Wrapper ========== #
class GeminiLLM(LLM):
    model: str = "models/gemini-1.5-flash"
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 512

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        return response.text.strip()

    @property
    def _llm_type(self) -> str:
        return "google-gemini"

# ========== 6. Embedding Helper ========== #
def get_gemini_embedding(text: str) -> list:
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="semantic_similarity"
    )
    return response["embedding"]

# ========== 7. Gemini-based Intent Classification ========== #
def classify_query_type(query: str) -> str:
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    prompt = f"""
You are a classifier that labels user queries as either 'keyword', 'semantic', or 'hybrid'.

Label as 'keyword' if the query:
- Refers only to specific fields (like title, author, brand, type, date_of_publish)
- Uses terms like “created by”, “greater than”, “before”, “between”, etc.

Label as 'semantic' if the query:
- Asks conceptual or natural questions (like “symptoms of depression”, “how to treat infection”)
- Has no obvious fields or structure

Label as 'hybrid' if the query:
- Combines field-based filters with conceptual or vague language
- Example: “procedures by author1 about infection”

Examples:

Query: documents created by user1 in the last 30 days  
Label: keyword

Query: symptoms of depression  
Label: semantic

Query: procedures by author1 about infection  
Label: hybrid

Query: show documents of type procedures authored by author1  
Label: keyword

Query: symptoms of depression  
Label: semantic

Query: how to cure flu  
Label: semantic

Query:Show documents about covid-19 by author2
Label:Hybrid

Now classify this query:

Query: {query}  
Label:
""".strip()
    response = model.generate_content(prompt)
    label = response.text.strip().lower()
    if "semantic" in label:
        return "semantic"
    elif "hybrid" in label:
        return "hybrid"
    else:
        return "keyword"

# ========== 8. Gemini Relevance Scoring Helper ========== #
def score_with_gemini(query: str, document_text: str) -> float:
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    prompt = f"""
Score the relevance of the following document to the query on a scale from 0 to 1.
Query: {query}
Document: {document_text}
Score only the number, no explanation.
"""
    response = model.generate_content(prompt)
    try:
        score = float(response.text.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.0

# ========== 9. LangChain Setup ========== #
llm = GeminiLLM()
prompt = get_few_shot_prompt()
chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())

# ========== 10. Streamlit UI ========== #
st.set_page_config(page_title="Solr + Gemini Semantic Search", layout="centered")
st.title("🔍 Solr + Gemini: Smart Search Assistant")

st.markdown("Enter a natural language query. The app will auto-detect if it should use keyword, semantic, or hybrid search.")

with st.expander("🧾 View Solr Fields"):
    st.code(", ".join(solr_fields))

raw_query = st.text_input("💬 Your Query:", placeholder="e.g. Show documents of type procedures OR Symptoms of depression")
user_query = expand_synonyms(raw_query)

if st.button("Generate & Search") and user_query.strip():
    with st.spinner("Detecting intent with Gemini and querying Solr..."):
        try:
            query_type = classify_query_type(user_query).capitalize()
            st.info(f"🔍 Detected Query Type (via Gemini): **{query_type} Search**")

            if query_type == "Keyword":
                solr_query = chain.run({
                    "user_query": user_query,
                    "fields": ", ".join(solr_fields)
                }).strip()
                st.success("✅ Generated Solr Query:")
                st.code(solr_query)
                results = solr.search(solr_query)

            elif query_type == "Semantic":
                embedding = get_gemini_embedding(user_query)
                vector_str = "[" + ",".join(map(str, embedding)) + "]"
                solr_query = f"{{!knn f=content_embedding topK=5}}{vector_str}"
                st.success("✅ Semantic Vector Query:")
                st.code(solr_query)
                results = solr.search(solr_query)

                scored_docs = []
                for doc in results:
                    doc_text = f"{doc.get('title', '')} {doc.get('content_text', '')}"
                    score = score_with_gemini(user_query, doc_text)
                    if score > 0.7:
                        filtered_doc = {field: doc.get(field, "") for field in solr_fields}
                        filtered_doc["score"] = score
                        scored_docs.append(filtered_doc)

                scored_docs.sort(key=lambda x: x["score"], reverse=True)
                results = scored_docs

            elif query_type == "Hybrid":
                keyword_query = chain.run({
                    "user_query": user_query,
                    "fields": ", ".join(solr_fields)
                }).strip()
                st.success("✅ Keyword Filter Query:")
                st.code(keyword_query)

                embedding = get_gemini_embedding(user_query)
                vector_str = "[" + ",".join(map(str, embedding)) + "]"
                semantic_query = f"{{!knn f=content_embedding topK=10}}{vector_str}"
                st.success("✅ Semantic Vector Query:")
                st.code(semantic_query)

                hybrid_query = f"{keyword_query} AND {semantic_query}"
                st.success("✅ Hybrid Query:")
                st.code(hybrid_query)

                results = solr.search(hybrid_query)
                scored_docs = []
                for doc in results:
                    doc_text = f"{doc.get('title', '')} {doc.get('content_text', '')}"
                    score = score_with_gemini(user_query, doc_text)
                    if score > 0.6:
                        filtered_doc = {field: doc.get(field, "") for field in solr_fields}
                        filtered_doc["score"] = score
                        scored_docs.append(filtered_doc)

                scored_docs.sort(key=lambda x: x["score"], reverse=True)
                results = scored_docs

            st.markdown(f"### 📄 Found {len(results)} result(s):")
            if results:
                st.dataframe(results, use_container_width=True)
            else:
                st.warning("No results found.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
