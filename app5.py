import streamlit as st
import google.generativeai as genai
import pysolr
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from prompt_template import get_few_shot_prompt
from langchain.llms.base import LLM
from typing import Optional, List, Dict, Tuple
import math

# ========== 1. Configure Gemini ========== #
genai.configure(api_key="AIzaSyDq2P1TXEzyBVHSc32FhsTDiwcR-qE25YM")  # 🔁 Replace with your Gemini API key

# ========== 2. Configure Solr ========== #
solr = pysolr.Solr("http://localhost:8983/solr/core4", always_commit=True, timeout=10)

# ========== 3. Solr Fields ========== #
solr_fields = ["id", "title", "content_text", "author", "brand", "type", "date_of_publish"]

# Sensible defaults for edismax hybrid fallback (tune for your schema)
EDISMAX_QF = "title^5 content_text^2 author^0.5 brand^0.5 type^0.2"
EDISMAX_PF = "title^8 content_text^3"
RECENCY_BF = "recip(ms(NOW,date_of_publish),3.16e-11,1,1)"  # light recency boost
ROWS = 25
KNN_TOPK_PRIMARY = 5
KNN_TOPK_FALLBACK = 75  # broaden on fallback

# ========== 4. Gemini LLM Wrapper ========== #
class GeminiLLM(LLM):
    model: str = "models/gemini-1.5-flash"
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 512

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        return (response.text or "").strip()

    @property
    def _llm_type(self) -> str:
        return "google-gemini"

# ========== 5. Embedding Helper ========== #
def get_gemini_embedding(text: str) -> List[float]:
    """Returns a list[float] embedding compatible with Solr's dense vector field."""
    # Gemini embeddings API
    emb = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")
    return emb["embedding"]

def average_vectors(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    length = len(vectors[0])
    sums = [0.0] * length
    for v in vectors:
        if len(v) != length:
            continue  # skip mis-sized vectors defensively
        for i, x in enumerate(v):
            sums[i] += float(x)
    n = max(1, len(vectors))
    return [s / n for s in sums]

# ========== 6. Gemini-based Intent Classification ========== #
def classify_query_type(query: str) -> str:
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    prompt = f"""
You are a classifier that labels user queries as either 'keyword' or 'semantic'.

Label as 'keyword' if the query:
- Refers to specific fields (like title, author, brand, type, date_of_publish)
- Uses terms like “created by”, “greater than”, “before”, “between”, etc.
- Involves filtering based on fields or values

Label as 'semantic' if the query:
- Asks conceptual or natural questions (like “symptoms of depression”, “how to treat infection”)
- Has no obvious fields or structure
- Is vague or short with a broad intent

Query: {query}
Label:
""".strip()
    response = model.generate_content(prompt)
    label = (response.text or "").strip().lower()
    return "semantic" if "semantic" in label else "keyword"

# ========== 7. LangChain Setup (for keyword query authoring) ========== #
llm = GeminiLLM()
prompt = get_few_shot_prompt()
chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())

# ========== 8. Hybrid Fallback Helpers ========== #
def expand_query_with_gemini(query: str) -> Dict[str, List[str]]:
    """
    Ask Gemini for expansions we can OR into an edismax query and also
    leverage for vector search. Keep output simple and parseable.
    """
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    p = f"""
Given the user query:
"{query}"

1) Provide up to 5 short keyphrases (≤3 words) capturing the intent.
2) Provide up to 3 paraphrases (concise).
3) Provide up to 6 domain-related synonyms/aliases.

Return as JSON with keys: keyphrases, paraphrases, synonyms. No extra text.
"""
    resp = model.generate_content(p)
    import json
    try:
        data = json.loads((resp.text or "").strip())
        return {
            "keyphrases": [s.strip() for s in data.get("keyphrases", []) if s.strip()],
            "paraphrases": [s.strip() for s in data.get("paraphrases", []) if s.strip()],
            "synonyms": [s.strip() for s in data.get("synonyms", []) if s.strip()],
        }
    except Exception:
        # very defensive fallback
        return {"keyphrases": [query], "paraphrases": [], "synonyms": []}

def build_edismax_query(user_query: str, exp: Dict[str, List[str]]) -> Tuple[str, Dict[str, str]]:
    """
    Build a broad, recall-friendly edismax query for fallback.
    - OR together: original, paraphrases, keyphrases, synonyms
    - Keep phrases quoted to benefit 'pf' boosting
    """
    terms = [user_query] + exp.get("paraphrases", []) + exp.get("keyphrases", []) + exp.get("synonyms", [])
    # Quote multi-word terms to help phrase matching; single tokens can be bare
    def fmt(t: str) -> str:
        t = t.strip()
        return f'"{t}"' if " " in t else t
    or_query = " OR ".join(fmt(t) for t in dict.fromkeys(terms))  # de-duplicate in order
    params = {
        "defType": "edismax",
        "qf": EDISMAX_QF,
        "pf": EDISMAX_PF,
        "ps": "2",
        "qs": "1",
        "q.op": "OR",
        "mm": "1",  # allow very permissive matching for recall
        "tie": "0.1",
        "bf": RECENCY_BF,
        "rows": str(ROWS * 3)  # broaden candidate pool for fusion
    }
    return or_query, params

def run_keyword(solr_q: str, params: Dict[str, str]):
    return solr.search(solr_q, **params)

def run_knn(vector: List[float], topk: int):
    vec = "[" + ",".join(map(str, vector)) + "]"
    knn_q = f"{{!knn f=content_embedding topK={topk}}}{vec}"
    return solr.search(knn_q, rows=str(min(topk, ROWS * 3)))

def rrf_fuse(bm25_docs: List[dict], knn_docs: List[dict], k: int = 60, max_out: int = ROWS) -> List[dict]:
    """
    Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank_d_list))
    """
    id_to_doc: Dict[str, dict] = {}
    ranks: Dict[str, float] = {}

    def add_list(docs: List[dict], weight: float = 1.0):
        for idx, d in enumerate(docs):
            doc_id = d.get("id")
            if not doc_id:
                continue
            id_to_doc.setdefault(doc_id, d)
            score = 1.0 / (k + (idx + 1))
            ranks[doc_id] = ranks.get(doc_id, 0.0) + weight * score

    add_list(bm25_docs, weight=1.0)
    add_list(knn_docs, weight=1.0)

    sorted_ids = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    fused = [id_to_doc[_id] for _id, _ in sorted_ids[:max_out]]
    return fused

def safe_table_rows(docs: List[dict]) -> List[dict]:
    return [{field: doc.get(field, "") for field in solr_fields} for doc in docs]

# ========== 9. Streamlit UI ========== #
st.set_page_config(page_title="Solr + Gemini Semantic Search", layout="centered")
st.title("🔍 Solr + Gemini: Smart Search Assistant")

st.markdown("Enter a natural language query. The app auto-detects keyword vs semantic; if both fail, it uses a hybrid fallback (query expansion + RRF fusion).")

with st.expander("🧾 View Solr Fields"):
    st.code(", ".join(solr_fields))

user_query = st.text_input("💬 Your Query:", placeholder="e.g. Show documents of type procedures OR Symptoms of depression")

if st.button("Generate & Search") and user_query.strip():
    with st.spinner("Detecting intent with Gemini and querying Solr..."):
        try:
            # ---------- Primary path: detect + run ----------
            query_type = classify_query_type(user_query).capitalize()
            st.info(f"🔍 Detected Query Type (via Gemini): **{query_type} Search**")

            results_primary = []
            results_semantic = []

            if query_type == "Keyword":
                # Author a structured keyword query via your few-shot prompt
                solr_query = chain.run({
                    "user_query": user_query,
                    "fields": ", ".join(solr_fields)
                }).strip()
                st.success("✅ Generated Solr (Keyword) Query:")
                st.code(solr_query)
                results_primary = list(solr.search(solr_query, rows=str(ROWS)))
            else:
                embedding = get_gemini_embedding(user_query)
                vector_str = "[" + ",".join(map(str, embedding)) + "]"
                solr_query = f"{{!knn f=content_embedding topK={KNN_TOPK_PRIMARY}}}{vector_str}"
                st.success("✅ Semantic Vector Query:")
                st.code(solr_query)
                results_semantic = list(solr.search(solr_query, rows=str(ROWS)))

            # ---------- Secondary path: try the other flavor as well ----------
            # If initial path returned 0, try the other type once before fallback.
            if not results_primary and query_type == "Semantic":
                # Try keyword pass via few-shot generator
                try:
                    solr_query_kw = chain.run({
                        "user_query": user_query,
                        "fields": ", ".join(solr_fields)
                    }).strip()
                    st.info("🧭 Also tried Keyword (edismax) query:")
                    st.code(solr_query_kw)
                    results_primary = list(solr.search(solr_query_kw, rows=str(ROWS)))
                except Exception:
                    pass

            if not results_semantic and query_type == "Keyword":
                try:
                    emb2 = get_gemini_embedding(user_query)
                    knn_q2 = f"{{!knn f=content_embedding topK={KNN_TOPK_PRIMARY}}}[" + ",".join(map(str, emb2)) + "]"
                    st.info("🧭 Also tried Semantic (knn) query:")
                    st.code(knn_q2)
                    results_semantic = list(solr.search(knn_q2, rows=str(ROWS)))
                except Exception:
                    pass

            # ---------- If either produced results, show best available ----------
            combined_now = (results_primary or []) + (results_semantic or [])
            if combined_now:
                st.markdown(f"### 📄 Found {len(combined_now)} result(s):")
                st.dataframe(safe_table_rows(combined_now[:ROWS]), use_container_width=True)
                st.stop()

            # ---------- Hybrid Fallback: Expansion + Broad EDisMax + Larger kNN + RRF ----------
            st.warning("No results from direct keyword or semantic search. Running hybrid fallback (expansion + fusion).")

            expansions = expand_query_with_gemini(user_query)
            st.write("**🔧 Expansions used**")
            st.json(expansions)

            # Build generous edismax query
            edismax_q, edismax_params = build_edismax_query(user_query, expansions)
            st.info("🧪 Fallback EDisMax Query:")
            st.code(f"q={edismax_q}\nparams={edismax_params}")

            # Run broad keyword search
            bm25_docs = list(run_keyword(edismax_q, edismax_params))

            # Compute an averaged embedding from original + paraphrases/keyphrases
            texts_for_vec = [user_query] + expansions.get("paraphrases", []) + expansions.get("keyphrases", [])
            vecs = []
            for t in texts_for_vec[:8]:  # cap calls
                try:
                    vecs.append(get_gemini_embedding(t))
                except Exception:
                    pass
            hybrid_vec = average_vectors(vecs) if vecs else get_gemini_embedding(user_query)

            # Run larger kNN
            knn_docs = list(run_knn(hybrid_vec, topk=KNN_TOPK_FALLBACK))

            # Fuse with RRF
            fused = rrf_fuse(bm25_docs, knn_docs, k=60, max_out=ROWS)

            st.markdown(f"### 📄 Hybrid results (RRF fused): {len(fused)}")
            if fused:
                st.dataframe(safe_table_rows(fused), use_container_width=True)
            else:
                # Last-resort ultra-broad fuzzy query (kept lightweight)
                fuzzy_q = f'"{user_query}"~3 {user_query}~2'
                st.info("🔎 Last-resort fuzzy EDisMax query (very broad):")
                st.code(fuzzy_q)
                fuzzy_docs = list(solr.search(
                    fuzzy_q,
                    defType="edismax",
                    qf=EDISMAX_QF,
                    pf=EDISMAX_PF,
                    mm="1",
                    q__op="OR",
                    rows=str(ROWS)
                ))
                if fuzzy_docs:
                    st.dataframe(safe_table_rows(fuzzy_docs), use_container_width=True)
                else:
                    st.warning("No results found, even after hybrid fallback.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
