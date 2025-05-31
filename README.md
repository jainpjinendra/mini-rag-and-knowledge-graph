# mini-rag-and-knowledge-graph

# AI-Assessment

## Assessment 1: Mini RAG Pipeline with ChromaDB on Colab

### Instructions:
1. Use Google Colab for the task.
2. Take 3–5 short documents (you can hardcode them).
3. Preprocess the text by converting to lowercase and removing punctuation.
4. **Embed the documents** using any method you prefer (e.g., `sentence-transformers`, HuggingFace, or other embedding techniques).
5. Store the embeddings in **ChromaDB** or **FAISS** (use ChromaDB for simplicity).
6. Accept a user query and retrieve the most relevant document based on cosine similarity.
7. Generate a simple response by outputting the most relevant document.

---

## Assessment 2: Building a Knowledge Graph with Semantic Search and Query Answering

### Instructions:
1. Build a basic **RDF knowledge graph** using the `rdflib` library.
   - Example: "Alice is a person," "Alice knows Bob," "Bob works at Acme Corp."
2. Convert the RDF triples into simple text sentences (e.g., "Alice knows Bob").
3. **Embed these text sentences** using any method you prefer (e.g., `sentence-transformers`, HuggingFace, or other embedding techniques).
4. Store the embeddings in **FAISS** or **ChromaDB**.
5. Accept a simple user query like: *"Who does Alice know?"* and return the relevant text from the stored embeddings.
