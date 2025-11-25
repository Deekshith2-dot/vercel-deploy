from sentence_transformers import SentenceTransformer, util

# load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# sentences to compare
s1 = "Flutter is a UI toolkit for building apps."
s2 = "React Native is used to build mobile apps."
s3 = "Virat Kohli is an Indian cricketer."

# generate embeddings
e1 = model.encode(s1)
e2 = model.encode(s2)
e3 = model.encode(s3)

# compute similarity
sim_1_2 = util.cos_sim(e1, e2)  # Flutter vs React Native
sim_1_3 = util.cos_sim(e1, e3)  # Flutter vs Virat Kohli

print("\nSimilarity Flutter vs React Native:", sim_1_2.item())
print("Similarity Flutter vs Virat Kohli:", sim_1_3.item())
