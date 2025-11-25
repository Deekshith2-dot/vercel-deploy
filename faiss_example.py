import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Our text database (3 chunks)
documents = [
    "Flutter is a UI toolkit for building natively compiled applications.",
    "React Native lets you create mobile apps using JavaScript.",
    "Virat Kohli is an Indian cricketer and former captain."
]

# 3. Convert all documents to embeddings
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype('float32')

# 4. Create FAISS index (vector database)
dimension = embeddings.shape[1]  # 384 dimensions
index = faiss.IndexFlatL2(dimension)

# 5. Add embeddings to the index
index.add(embeddings)

# 6. User query
query = "How can I build mobile apps?"
query_embedding = model.encode([query]).astype('float32')

# 7. Search in FAISS (top-2 results)
k = 2
distances, indices = index.search(query_embedding, k)

# 8. Print results
print("Query:", query)
print("\nTop matches:")
for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1} → {documents[idx]}")
    print(f"Distance: {distances[0][i]}")
