from rdflib import Graph, Namespace, RDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Build RDF Graph
g = Graph()
EX = Namespace("http://example.org/")
g.add((EX.Alice, RDF.type, EX.Person))
g.add((EX.Alice, EX.knows, EX.Bob))
g.add((EX.Bob, RDF.type, EX.Person))
g.add((EX.Bob, EX.worksAt, EX.AcmeCorp))
g.add((EX.AcmeCorp, RDF.type, EX.Organization))

# Step 2: Convert Triples to Sentences
triples = [f"{s.split('/')[-1]} {p.split('/')[-1]} {o.split('/')[-1]}" for s, p, o in g]

# Step 3: Embed Sentences
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(triples)

# Step 4: Store Embeddings in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 5: Query Handling
query = "Who does Alice know?"
query_embedding = model.encode([query])
D, I = index.search(np.array(query_embedding), k=3)

# Display Results
print(f"User Query: {query}")
for idx in I[0]:
    print(f"Matched: {triples[idx]}")