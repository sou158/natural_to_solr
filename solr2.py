import os
import uuid
import sys
import pdfplumber
import docx
import pysolr
from datetime import datetime
from openai import AzureOpenAI

# ========== 1. Configure Azure OpenAI ========== #
# Make sure these environment variables are set or replace them with hardcoded values
AZURE_OPENAI_ENDPOINT =  "https://azdtapimanager.azure-api.net/newllm/deployments/dt_trial_text-embedding-3-large/embeddings?api-version=2023-05-15"
AZURE_OPENAI_KEY =  "280ea43fe4674b42adfaa2bddbe45d9f"
AZURE_OPENAI_DEPLOYMENT = "text-embedding-3-large"
# Replace with your deployment name (for text-embedding-3-small or text-embedding-3-large)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ========== 2. Solr Setup ========== #
solr = pysolr.Solr("http://localhost:8983/solr/core6", always_commit=True, timeout=10)


# ========== 3. Extract Text ========== #
def extract_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_path.lower().endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type")


# ========== 4. Get Embedding from Azure OpenAI ========== #
def get_embedding(text: str) -> list:
    # Truncate text to avoid exceeding token limits
    truncated_text = text[:3000]

    response = client.embeddings.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        input=truncated_text
    )

    return response.data[0].embedding


# ========== 5. Upload a Single File with Extra Fields ========== #
def upload_document(file_path: str):
    print(f"📄 Processing: {file_path}")
    try:
        content_text = extract_text(file_path)
        if not content_text.strip():
            print("⚠️ Skipped: No extractable text.")
            return

        embedding = get_embedding(content_text)

        # --- Manually enter metadata fields ---
        author = input("Enter Author name (or leave blank): ").strip() or "Unknown"
        brand = input("Enter Brand (or leave blank): ").strip() or "Unknown"
        doc_type = input("Enter Document Type (or leave blank): ").strip() or "General"
        date_input = input("Enter Date of Publish (YYYY-MM-DD) (or leave blank for today): ").strip()
        if date_input:
            try:
                date_of_publish = datetime.strptime(date_input, "%Y-%m-%d").strftime("%Y-%m-%dT00:00:00Z")
            except ValueError:
                print("⚠️ Invalid date format. Using today's date instead.")
                date_of_publish = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")
        else:
            date_of_publish = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")

        # --- Solr Document ---
        doc = {
            "id": str(uuid.uuid4()),
            "title": os.path.basename(file_path),
            "content_text": content_text,
            "content_embedding": embedding,
            "author": author,
            "brand": brand,
            "type": doc_type,
            "date_of_publish": date_of_publish
        }

        solr.add([doc])
        print("✅ Uploaded with additional metadata.")

    except Exception as e:
        print(f"❌ Failed to upload {file_path}: {e}")


# ========== 6. Entry Point ========== #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python upload_folder_to_solr.py path/to/file")
        sys.exit(1)

    upload_document(sys.argv[1])
