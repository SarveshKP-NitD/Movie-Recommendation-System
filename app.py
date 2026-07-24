# ============================================================
# Movie Recommendation System using Neural Collaborative Filtering
# Developed by Sarvesh Kumar Pal
# M.Tech CSE | NIT Delhi
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.subheader("Neural Collaborative Filtering (NCF) based Personalized Recommendation")

# ============================================================
# Cache Dataset
# ============================================================

@st.cache_data
def load_dataset():

    ratings = pd.read_csv(
        "data/ml-100k/u.data",
        sep="\t",
        names=[
            "user_id",
            "movie_id",
            "rating",
            "timestamp"
        ]
    )

    movies = pd.read_csv(
        "data/ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0,1],
        names=[
            "movie_id",
            "movie_title"
        ]
    )

    ratings = ratings.merge(
        movies,
        on="movie_id"
    )

    return ratings


ratings = load_dataset()

st.success("Dataset Loaded Successfully!")

# ============================================================
# Dataset Preview
# ============================================================

with st.expander("Dataset Preview"):

    st.dataframe(
        ratings.head(),
        use_container_width=True
    )

# ============================================================
# Dataset Statistics
# ============================================================

total_users = ratings.user_id.nunique()
total_movies = ratings.movie_id.nunique()
total_ratings = len(ratings)

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Project Information")

st.sidebar.write(f"👤 Users : {total_users}")
st.sidebar.write(f"🎬 Movies : {total_movies}")
st.sidebar.write(f"⭐ Ratings : {total_ratings}")

st.sidebar.markdown("---")

# ============================================================
# Create Encoders
# ============================================================

user_ids = ratings["user_id"].unique()
movie_ids = ratings["movie_id"].unique()

user2encoded = {
    x:i
    for i,x in enumerate(user_ids)
}

encoded2user = {
    i:x
    for i,x in enumerate(user_ids)
}

movie2encoded = {
    x:i
    for i,x in enumerate(movie_ids)
}

encoded2movie = {
    i:x
    for i,x in enumerate(movie_ids)
}

ratings["user_encoded"] = ratings.user_id.map(user2encoded)

ratings["movie_encoded"] = ratings.movie_id.map(movie2encoded)

st.success("Encoding Created Successfully!")
# ============================================================
# NCF Model
# ============================================================

class NCF(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=50
    ):

        super(NCF, self).__init__()

        # User Embedding
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        # Movie Embedding
        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(
            embedding_dim * 2,
            128
        )

        self.fc2 = nn.Linear(
            128,
            64
        )

        self.output = nn.Linear(
            64,
            1
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.2)

    def forward(
        self,
        user,
        movie
    ):

        user_vector = self.user_embedding(user)

        movie_vector = self.movie_embedding(movie)

        x = torch.cat(
            [user_vector, movie_vector],
            dim=1
        )

        x = self.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.relu(self.fc2(x))

        prediction = self.output(x)

        return prediction.squeeze()


# ============================================================
# Load Trained Model
# ============================================================

@st.cache_resource
def load_model():

    model = NCF(
        len(user2encoded),
        len(movie2encoded)
    )

    model.load_state_dict(
        torch.load(
            "models/ncf_model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


model = load_model()

st.success("✅ Trained NCF Model Loaded Successfully!")
# ============================================================
# Recommendation Function
# ============================================================

def recommend_movies(user_id, top_n=10):

    # Check if user exists
    if user_id not in user2encoded:

        st.error("User ID not found!")

        return pd.DataFrame()

    # Get encoded user
    encoded_user = user2encoded[user_id]

    # Movies already watched
    watched_movies = ratings[
        ratings["user_id"] == user_id
    ]["movie_id"].tolist()

    # Candidate movies (not watched)
    candidate_movies = ratings[
        ~ratings["movie_id"].isin(watched_movies)
    ][["movie_id", "movie_title"]]

    candidate_movies = candidate_movies.drop_duplicates()

    predictions = []

    # Disable gradient computation
    with torch.no_grad():

        for _, row in candidate_movies.iterrows():

            movie_id = row["movie_id"]

            movie_title = row["movie_title"]

            # Skip if movie encoding not found
            if movie_id not in movie2encoded:
                continue

            encoded_movie = movie2encoded[movie_id]

            user_tensor = torch.tensor(
                [encoded_user],
                dtype=torch.long
            )

            movie_tensor = torch.tensor(
                [encoded_movie],
                dtype=torch.long
            )

            predicted_rating = model(
                user_tensor,
                movie_tensor
            ).item()

            predictions.append(
                {
                    "movie_id": movie_id,
                    "movie_title": movie_title,
                    "predicted_rating": predicted_rating
                }
            )

    # Convert to DataFrame
    recommendations = pd.DataFrame(predictions)

    # Highest predicted rating first
    recommendations = recommendations.sort_values(
        by="predicted_rating",
        ascending=False
    )

    recommendations.reset_index(
        drop=True,
        inplace=True
    )

    return recommendations.head(top_n)


# ============================================================
# Sidebar Controls
# ============================================================

st.sidebar.header("Recommendation Settings")

selected_user = st.sidebar.selectbox(

    "Select User ID",

    sorted(ratings["user_id"].unique())

)

top_n = st.sidebar.slider(

    "Number of Recommendations",

    min_value=5,

    max_value=20,

    value=10,

    step=1

)

st.sidebar.markdown("---")
# ============================================================
# Main Recommendation Interface
# ============================================================

st.markdown("---")

st.header("🎯 Get Personalized Movie Recommendations")

st.write(
    """
Select a **User ID** from the sidebar and click the button below
to generate personalized movie recommendations using the trained
Neural Collaborative Filtering (NCF) model.
"""
)

# ============================================================
# Recommendation Button
# ============================================================

if st.button("🎬 Recommend Movies"):

    with st.spinner("Generating recommendations..."):

        recommendations = recommend_movies(
            selected_user,
            top_n
        )

    st.success("Recommendations Generated Successfully!")

    # ========================================================
    # Display Table
    # ========================================================

    st.subheader(f"Top {top_n} Recommended Movies")

    st.dataframe(
        recommendations,
        use_container_width=True
    )

    # ========================================================
    # Movie Cards
    # ========================================================

    st.markdown("---")

    st.subheader("🎥 Recommendation Details")

    for index, row in recommendations.iterrows():

        with st.container():

            st.markdown(
                f"""
### {index+1}. {row['movie_title']}

⭐ **Predicted Rating:** `{row['predicted_rating']:.3f}`

Movie ID : `{row['movie_id']}`

---
"""
            )

# ============================================================
# Footer
# ============================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Users",
        total_users
    )

with col2:

    st.metric(
        "Movies",
        total_movies
    )

with col3:

    st.metric(
        "Ratings",
        total_ratings
    )

st.markdown("---")

st.markdown(
"""
## 📌 Project Summary

This application demonstrates a **Movie Recommendation System**
built using **Neural Collaborative Filtering (NCF)**.

### 🚀 Features

- Personalized Recommendations
- Deep Learning using PyTorch
- Neural Collaborative Filtering
- MovieLens 100K Dataset
- Interactive Streamlit Interface

---

### 🛠 Tech Stack

- Python
- PyTorch
- Streamlit
- Pandas
- NumPy
- MovieLens 100K

---

### 👨‍💻 Developed By

**Sarvesh Kumar Pal**

M.Tech (Computer Science & Engineering)

National Institute of Technology Delhi

GitHub:
https://github.com/SarveshKP-NitD

Year: **2026**
"""
)