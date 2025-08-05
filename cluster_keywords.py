from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd

# Step 1: Add your keyword list
keywords = [
    "buy red shoes",
    "cheap sneakers",
    "Adidas women’s shoes",
    "Nike running shoes",
    "men’s formal shoes",
    "women’s sandals",
    "sports shoes for kids",
    "high heels for party"
]

# Step 2: Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Convert keywords to vector embeddings
embeddings = model.encode(keywords)

# Step 4: Choose number of clusters (you can tune this)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)
labels = kmeans.labels_

# Step 5: Output results
df = pd.DataFrame({'Keyword': keywords, 'Cluster': labels})
# Group and display the clusters
grouped = df.groupby('Cluster')

for cluster_num, group in grouped:
    print(f"\n--- Cluster {cluster_num + 1} ---")
    for keyword in group['Keyword']:
        print(f"- {keyword}")

# Optional: Save to CSV
df.to_csv("clustered_keywords.csv", index=False)