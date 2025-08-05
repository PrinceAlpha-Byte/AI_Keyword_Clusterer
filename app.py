import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# UI Title
st.title("AI Keyword Clusterer")
st.write("Upload a CSV file with a column of keywords, and this AI will group them into clusters for you.")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'Keyword' not in df.columns:
        st.error("The CSV must contain a 'Keyword' column.")
    else:
        keywords = df['Keyword'].dropna().tolist()

        # Get embeddings
        embeddings = model.encode(keywords)

        # Choose number of clusters
        num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        clusters = kmeans.fit_predict(embeddings)

        # Add to DataFrame
        df['Cluster'] = clusters

        # Display Results
        st.subheader("Clustered Keywords:")
        for cluster_num in range(num_clusters):
            st.markdown(f"### Cluster {cluster_num + 1}")
            cluster_keywords = df[df['Cluster'] == cluster_num]['Keyword'].tolist()
            for kw in cluster_keywords:
                st.write(f"- {kw}")

        # Allow download
        csv = df.to_csv(index=False)
        st.download_button("Download Clustered CSV", csv, "clustered_keywords.csv", "text/csv")