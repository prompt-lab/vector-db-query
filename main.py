import openai
from pymilvus import connections, Collection

# Set your OpenAI API key
openai.api_key = ''

# Connect to Milvus server
connections.connect("default", host="localhost", port="19530")

# Get the collection
collection = Collection("OpenAI_Embeddings")

# Embed the query string using OpenAI
response = openai.Embedding.create(
    input="second",
    model="text-embedding-ada-002"
)
query_embedding = response["data"][0]["embedding"]

# Load the collection to memory before performing a search
collection.load()

# Perform a vector similarity search
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}
result = collection.search([query_embedding], "embedding", search_params, limit=1, output_fields=["original_string"])

# The result variable now contains the search results. You can print them out or use them in your application.
print(result)
