# Import libraries
import chromadb
from sentence_transformers import SentenceTransformer
import string

# Initialize ChromaDB in-memory
client = chromadb.Client()
collection = client.create_collection("documents")

# Hardcode 3-5 short documents
documents = [
    "The sun sets beautifully over the ocean, casting a golden hue.",
    "Artificial intelligence is transforming industries across the globe.",
    "In the world of physics, quantum mechanics describes subatomic particles.",
    "Healthy eating and regular exercise improve overall well-being.",
    "Space exploration has expanded human understanding of the universe."
]

# Preprocess text: lowercase and remove punctuation
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

preprocessed_docs = [preprocess(doc) for doc in documents]

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(preprocessed_docs)

# Add documents and embeddings to ChromaDB
for i, (doc, embedding) in enumerate(zip(preprocessed_docs, embeddings)):
    collection.add(
        ids=[f"doc_{i}"],
        embeddings=[embedding],
        documents=[doc]
    )

# Function to handle user query
def retrieve_relevant_doc(query):
    # Preprocess the query
    processed_query = preprocess(query)
    query_embedding = model.encode([processed_query])[0]
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )
    
    # Get the most relevant document
    if results["documents"]:
        return results["documents"][0][0]
    else:
        return "No relevant document found."

# Example usage
user_query = input("Enter your query: ")
response = retrieve_relevant_doc(user_query)
print("\nMost relevant document:\n")
print(response)