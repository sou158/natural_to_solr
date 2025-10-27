import streamlit as st
from openai import AzureOpenAI
import pysolr
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from prompt_template import get_few_shot_prompt
from langchain.llms.base import LLM
from typing import Optional, List
import json
import os

# ================= Configuration =================
AZURE_CHAT_MODEL = "dt_trial_gpt-4o"
AZURE_EMBED_MODEL = "dt_trial_text-embedding-3-large"

BASE_URL_embed = "https://azdtapimanager.azure-api.net/newllm/deployments/dt_trial_text-embedding-3-large/embeddings?api-version=2023-05-15"
BASE_URL_chat= "https://azdtapimanager.azure-api.net/newllm/deployments/dt_trial_gpt-4o/chat/completions?api-version=2024-08-01-preview"
AZURE_OPENAI_API_KEY = "280ea43fe4674b42adfaa2bddbe45d9f"

chat_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,                  
    azure_endpoint=BASE_URL_chat,
    api_version="2024-08-01-preview"
)

embed_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=BASE_URL_embed,
    api_version="2023-05-15"
)

# ========== 2. Configure Solr ========== #
solr = pysolr.Solr("http://localhost:8983/solr/core8", always_commit=True, timeout=10)

# ========== 3. Solr Fields ========== #
solr_fields = ["id", "title", "content_text", "author", "brand", "type", "date_of_publish", "content_embedding"]

# ========== 4. Synonym Expansion ========== #
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

# ========== Utility: normalize text fields (handle list vs str) ========== #

def normalize_text(value):
    """Convert possibly-list values from Solr into a single string."""
    if value is None:
        return ""
    if isinstance(value, list):
        # join list elements into a single string
        return " ".join([str(v) for v in value])
    return str(value)

# ========== 5. Azure OpenAI LLM Wrapper ========== #
class AzureOpenAILLM(LLM):
    model: str = AZURE_CHAT_MODEL
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 512

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # IMPORTANT: use the chat_client (not embed_client)
        response = chat_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p
        )
        # support both message.content or text depending on SDK
        try:
            return response.choices[0].message.content.strip()
        except Exception:
            # fallback if response shape differs
            return str(response).strip()

    @property
    def _llm_type(self) -> str:
        return "azure-openai"

# ========== 6. Embedding Helper ========== #
def get_azure_embedding(text: str) -> list:
    """Return Azure embedding vector as list of floats."""
    if not text:
        return []
    response = embed_client.embeddings.create(
        model=AZURE_EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

# ========== 7. Query Classification ========== #
def classify_query_type(query: str) -> str:
    prompt = f"""
You are a classifier that labels user queries as either 'keyword' or 'semantic'.

Label as 'keyword' if the query:
- Refers to specific fields (like title, author, brand, type, date_of_publish)
- Uses terms like "created by", "greater than", "before", "between", etc.
- Involves filtering based on fields or values

Label as 'semantic' if the query:
- Asks conceptual or natural questions (like "percentage of alcohol in sanitizers", "how to treat infections", "work ID for task A")
- Has no obvious fields or structure
- Is vague or short with a broad intent

Now classify this query:
Query: {query}
Label:
""".strip()

    response = chat_client.chat.completions.create(
        model=AZURE_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    label = response.choices[0].message.content.strip().lower()
    return "semantic" if "semantic" in label else "keyword"

# ========== 8. Relevance Scoring ========== #
def score_with_azure(query: str, document_text: str) -> float:
    prompt = f"""
Score the relevance of the following document to the query on a scale from 0 to 1.
Query: {query}
Document: {document_text}
Score only the number, no explanation.
""".strip()

    response = chat_client.chat.completions.create(
        model=AZURE_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        score = float(response.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.0

# ========== 9. LangChain Setup ========== #
llm = AzureOpenAILLM()
prompt = get_few_shot_prompt()
chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())

# ========== 10. Streamlit UI ========== #
st.set_page_config(page_title="Solr + Azure OpenAI Semantic Search", layout="centered")
st.title("🔍 Solr + Azure OpenAI: Smart Search Assistant")

st.markdown("Enter a natural language query. The app will auto-detect if it should use keyword or semantic search.")

with st.expander("🧾 View Solr Fields"):
    st.code(", ".join(solr_fields))

raw_query = st.text_input("💬 Your Query:", placeholder="e.g. Show documents of type procedures OR Symptoms of depression")
user_query = expand_synonyms(raw_query)

if st.button("Generate & Search") and user_query.strip():
    with st.spinner("Detecting intent with Azure OpenAI and querying Solr..."):
        try:
            query_type = classify_query_type(user_query).capitalize()
            st.info(f"🔍 Detected Query Type (via Azure OpenAI): **{query_type} Search**")

            results = []

            if query_type == "Keyword":
                # --- Generate and run Solr keyword query --- #
                solr_query = chain.run({
                    "user_query": user_query,
                    "fields": ", ".join(solr_fields)
                }).strip()
                if solr_query.lower().startswith("solr query:"):
                    solr_query = solr_query.split(":", 1)[1].strip()

                st.success("✅ Generated Solr Query:")
                st.code(solr_query)
                results = solr.search(q=solr_query)

            elif query_type == "Semantic":
                # --- Generate embedding and query Solr using KNN --- #
                embedding = get_azure_embedding(user_query)
                st.write("📏 Embedding length:", len(embedding))

                if not embedding:
                    st.warning("Failed to create embedding for query.")
                    results = []
                else:
                    # Convert embedding list to JSON array for Solr (as text)
                    vector_json = json.dumps(embedding)
                    solr_query = f"{{!knn f=content_embedding topK=5}}{vector_json}"

                    raw_results = solr.search(solr_query, **{
                        "fl": ",".join(solr_fields),
                        "rows": 5
                    })

                    # Normalize and re-score results
                    scored_docs = []
                    for doc in raw_results:
                        title_text = normalize_text(doc.get('title', ''))
                        content_text = normalize_text(doc.get('content_text', ''))
                        doc_text = f"{title_text} {content_text}".strip()

                        score = score_with_azure(user_query, doc_text)
                        if score > 0.7:
                            filtered_doc = {field: normalize_text(doc.get(field, "")) for field in solr_fields}
                            filtered_doc["score"] = score
                            scored_docs.append(filtered_doc)

                    scored_docs.sort(key=lambda x: x["score"], reverse=True)
                    results = scored_docs

                    # --- Generate final LLM answer using RAG --- #
                    if results:
                        context = "\n\n".join([normalize_text(doc.get("content_text")) for doc in results[:3]])
                        rag_prompt = f"""
You are an assistant. Answer the following question using only the provided documents.

Question: {user_query}

Documents:
{context}

Answer:
""".strip()

                        response = chat_client.chat.completions.create(
                            model=AZURE_CHAT_MODEL,
                            messages=[{"role": "user", "content": rag_prompt}],
                            temperature=0.3
                        )
                        rag_answer = response.choices[0].message.content.strip()
                        st.markdown("### 🤖 LLM Answer:")
                        st.write(rag_answer)

            # --- Display Results --- #
            st.markdown(f"### 📄 Found {len(results)} result(s):")
            if results:
                st.dataframe(results, use_container_width=True)
            else:
                st.warning("No results found.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
