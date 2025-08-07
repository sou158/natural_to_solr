from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

# New Solr Fields
solr_fields = ["id", "title", "content_text", "author", "brand", "type", "date_of_publish"]

# Few-shot examples for keyword query generation
examples = [
    {
        "user_query": 'documents that contain "Name1" in title',
        "solr_query": 'title:"Name1"'
    },
    {
        "user_query": 'documents created or written by "user 1"',
        "solr_query": 'author:"user 1"'
    },
    {
        "user_query": 'documents of type procedures created by author1 in last 30 days',
        "solr_query": 'type:"procedures" AND author:"author1" AND date_of_publish:[NOW-30DAYS TO NOW]'
    },
    {
        "user_query": 'find documents with brand "Pfizer"',
        "solr_query": 'brand:"Pfizer"'
    },
    {
        "user_query": 'get documents published before 2023-01-01',
        "solr_query": 'date_of_publish:[* TO 2023-01-01T00:00:00Z]'
    }
]

# Prompt template for generating Solr queries
example_prompt = PromptTemplate(
    input_variables=["user_query", "solr_query"],
    template="User Query: {user_query}\nSolr Query: {solr_query}"
)

# Final prompt with few-shot examples
def get_few_shot_prompt():
    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=(
            f"You are an expert Solr query builder. You take a user natural language question and generate an optimized Solr query "
            f"using ONLY the following fields: {', '.join(solr_fields)}.\n"
            f"Always use proper Solr syntax with AND/OR, field names, and range queries if needed.\n"
            f"Here are some examples:"
        ),
        suffix="User Query: {user_query}\nSolr Query:",
        input_variables=["user_query"],
    )
