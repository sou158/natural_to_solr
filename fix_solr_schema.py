import requests
import json

# ========== CONFIG ==========
SOLR_URL = "http://localhost:8983/solr/core8"  # Change 'core7' to your core name
FIELD_NAME = "content_embedding"
VECTOR_TYPE = "knn_vector"
VECTOR_DIM = 3072  # Use 1536 for text-embedding-3-small

headers = {"Content-Type": "application/json"}


def check_field_type():
    """Check or create knn_vector field type"""
    print(f"🔍 Checking if field type '{VECTOR_TYPE}' exists...")
    url = f"{SOLR_URL}/schema/fieldtypes/{VECTOR_TYPE}?wt=json"
    resp = requests.get(url)

    if resp.status_code == 200 and "fieldType" in resp.json():
        print(f"✅ Field type '{VECTOR_TYPE}' already exists.")
        return True
    else:
        print(f"⚠️ Field type '{VECTOR_TYPE}' not found. Creating it now...")
        payload = {
            "add-field-type": {
                "name": VECTOR_TYPE,
                "class": "solr.DenseVectorField",
                "vectorDimension": VECTOR_DIM,
                "similarityFunction": "cosine"
            }
        }
        add_type_resp = requests.post(f"{SOLR_URL}/schema", headers=headers, data=json.dumps(payload))
        if add_type_resp.status_code == 200:
            print(f"✅ Field type '{VECTOR_TYPE}' created successfully.")
            return True
        else:
            print("❌ Failed to create field type:", add_type_resp.text)
            return False


def check_and_fix_field():
    """Ensure the content_embedding field is correctly set up"""
    print(f"\n🔍 Checking if field '{FIELD_NAME}' exists...")
    check_url = f"{SOLR_URL}/schema/fields/{FIELD_NAME}?wt=json"
    response = requests.get(check_url)

    if response.status_code == 200 and "field" in response.json():
        existing_type = response.json()["field"]["type"]
        print(f"✅ Field '{FIELD_NAME}' exists with type: {existing_type}")
        if existing_type != VECTOR_TYPE:
            print("⚠️ Field type mismatch. Deleting and recreating...")
            del_response = requests.post(
                f"{SOLR_URL}/schema",
                headers=headers,
                data=json.dumps({"delete-field": {"name": FIELD_NAME}})
            )
            if del_response.status_code == 200:
                print("🗑️ Deleted old field successfully.")
            else:
                print("❌ Error deleting field:", del_response.text)
        else:
            print("✅ Field already correct. No action needed.")
            return
    else:
        print("ℹ️ Field not found. Creating new field...")

    # 🟢 FIXED: Removed vectorDimension and vectorSimilarityFunction from here
    add_field_payload = {
        "add-field": {
            "name": FIELD_NAME,
            "type": VECTOR_TYPE,
            "indexed": True,
            "stored": True
        }
    }

    add_response = requests.post(f"{SOLR_URL}/schema", headers=headers, data=json.dumps(add_field_payload))
    if add_response.status_code == 200:
        print(f"✅ Field '{FIELD_NAME}' created successfully!")
    else:
        print("❌ Error creating field:", add_response.text)

    verify = requests.get(check_url)
    print("\n🔍 Verification result:")
    print(json.dumps(verify.json(), indent=2))


if __name__ == "__main__":
    print("🚀 Starting Solr schema auto-fix...\n")
    if check_field_type():
        check_and_fix_field()
    else:
        print("❌ Could not verify or create vector field type. Exiting.")
