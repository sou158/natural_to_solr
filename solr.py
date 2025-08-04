import os
import uuid
import sys
import pdfplumber
from docx import Document
import pysolr
import google.generativeai as genai

# ========== 1. Configure Gemini ========== #
genai.configure(api_key="AIzaSyDq2P1TXEzyBVHSc32FhsTDiwcR-qE25YM")  # Replace with your actual Gemini API key

# ========== 2. Solr Setup ========== #
solr = pysolr.Solr("http://localhost:8983/solr/core1", always_commit=True, timeout=10)


# ========== 3. Extract Text ========== #
def extract_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type")

# ========== 4. Get Embedding from Gemini ========== #
def get_embedding(text: str) -> list:
    response = genai.embed_content(
        model="models/embedding-001",
        content=text[:3000],
        task_type="semantic_similarity"
    )
    # handle both possible formats
    embedding = response.get("embedding", [])
    if isinstance(embedding, dict) and "values" in embedding:
        return embedding["values"]
    return embedding



# ========== 5. Upload a Single File ========== #
def upload_document(file_path: str):
    print(f"📄 Processing: {file_path}")
    try:
        content_text = extract_text(file_path)
        if not content_text.strip():
            print("⚠️ Skipped: No extractable text.")
            return

        embedding = get_embedding(content_text)

        print(type(embedding), len(embedding))
        print(embedding[:10])

        doc = {
            "id": str(uuid.uuid4()),
            "title": os.path.basename(file_path),
            "content_text": content_text,
            "content_embedding": embedding
        }

        solr.add([doc])
        print("✅ Uploaded.")
    except Exception as e:
        print(f"❌ Failed to upload {file_path}: {e}")

# ========== 6. Upload All Files in Folder ========== #
# def upload_folder(folder_path: str):
#     if not os.path.isdir(folder_path):
#         print("❌ Invalid folder path.")
#         return

#     supported_ext = [".pdf", ".docx"]
#     files = [
#         os.path.join(folder_path, f)
#         for f in os.listdir(folder_path)
#         if f.lower().endswith(tuple(supported_ext))
#     ]

#     if not files:
#         print("📂 No supported files found in folder.")
#         return

#     print(f"📁 Found {len(files)} files. Uploading...")
#     for file_path in files:
#         upload_document(file_path)

# ========== 7. Entry Point ========== #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python upload_folder_to_solr.py path/to/folder")
        sys.exit(1)

    upload_document(sys.argv[1])
