# 🎬 Movie Recommendation System using Neural Collaborative Filtering (NCF)

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📌 Project Overview

This project presents an **end-to-end Movie Recommendation System** built using the **MovieLens 100K Dataset**. The recommendation engine combines **Deep Learning**, **Collaborative Filtering**, and **Vector Search** to generate personalized movie recommendations for users.

The system was developed as part of an exploration of modern recommender systems using **PyTorch**, **FAISS**, and **Streamlit**, demonstrating the complete machine learning workflow—from data preprocessing and model training to deployment through an interactive web application.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Build a personalized movie recommendation engine.
- Learn the fundamentals of recommendation systems.
- Implement Collaborative Filtering techniques.
- Develop a Neural Collaborative Filtering (NCF) model using PyTorch.
- Integrate FAISS for efficient vector similarity search.
- Deploy the trained model using Streamlit.
- Compare multiple recommendation approaches.
- Create a production-ready machine learning project for portfolio purposes.

---

# ✨ Features

- 🎬 Personalized Movie Recommendations
- 🤖 Neural Collaborative Filtering (NCF)
- 🔍 FAISS Vector Similarity Search
- 📊 Exploratory Data Analysis (EDA)
- 🧠 Learned User & Movie Embeddings
- ⚡ Fast Recommendation Generation
- 📈 Model Evaluation (RMSE & MAE)
- 🌐 Interactive Streamlit Web Application
- 📁 Well-Structured Modular Project
- 📚 Jupyter Notebook Documentation

---

# 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Vector Search | FAISS |
| Visualization | Matplotlib |
| Web Framework | Streamlit |
| Dataset | MovieLens 100K |

---

# 📂 Dataset

**Dataset Name:** MovieLens 100K

The MovieLens 100K dataset contains movie ratings collected by the GroupLens Research Project.

### Dataset Statistics

| Metric | Value |
|--------|------:|
| Users | 943 |
| Movies | 1,682 |
| Ratings | 100,000 |
| Rating Scale | 1–5 |

---

# 🏗️ Project Workflow

```
MovieLens Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ▼
User & Movie Encoding
        │
        ▼
Embedding Generation
        │
        ▼
FAISS Vector Index
        │
        ▼
Neural Collaborative Filtering
        │
        ▼
Model Evaluation
        │
        ▼
Movie Recommendation Engine
        │
        ▼
Streamlit Web Application
```

---

# 🧠 Models Implemented

This project explores multiple recommendation approaches.

| Model | Purpose |
|--------|----------|
| Popularity Based | Baseline Recommendation |
| User-Based Collaborative Filtering | Similar Users |
| Item-Based Collaborative Filtering | Similar Movies |
| Singular Value Decomposition (SVD) | Latent Factor Recommendation |
| FAISS Vector Search | Fast Similarity Search |
| Neural Collaborative Filtering (NCF) | Deep Learning Recommendation |
# 📁 Repository Structure

```text
Movie-Recommendation-System/
│
├── app.py                         # Streamlit Web Application
├── requirements.txt               # Project Dependencies
├── README.md                      # Project Documentation
│
├── data/
│   └── ml-100k/
│       ├── u.data
│       ├── u.item
│       ├── u.genre
│       ├── u.user
│       └── ...
│
├── models/
│   ├── ncf_model.pth
│   ├── user_faiss.index
│   ├── user_embeddings.csv
│   ├── item_embeddings.csv
│   ├── ncf_user_embeddings.csv
│   ├── ncf_movie_embeddings.csv
│   ├── user_mapping.csv
│   ├── item_mapping.csv
│   └── sample_recommendations.csv
│
├── notebooks/
│   ├── 01_Dataset_Loading.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_User_Item_Embeddings.ipynb
│   ├── 05_FAISS_Vector_Search.ipynb
│   ├── 06_User_Based_Recommendation.ipynb
│   ├── 07_Item_Based_Recommendation.ipynb
│   ├── 08_FAISS_Recommendation.ipynb
│   ├── 09_Neural_Collaborative_Filtering.ipynb
│   ├── 10_NCF_Recommendation.ipynb
│   ├── 11_Model_Evaluation.ipynb
│   ├── 12_Model_Comparison.ipynb
│   ├── 13_Streamlit_App_Development.ipynb
│   └── 14_Project_Conclusion.ipynb
│
├── images/
│   ├── app_home.png
│   ├── recommendation_output.png
│   ├── architecture.png
│   └── workflow.png
│
└── LICENSE
```

---

# 📒 Notebook Description

| Notebook | Description |
|-----------|-------------|
| Notebook 01 | Dataset Loading and Inspection |
| Notebook 02 | Data Cleaning & Preprocessing |
| Notebook 03 | Exploratory Data Analysis (EDA) |
| Notebook 04 | User & Movie Embedding Generation |
| Notebook 05 | FAISS Vector Index Creation |
| Notebook 06 | User-Based Recommendation |
| Notebook 07 | Item-Based Recommendation |
| Notebook 08 | FAISS Similar User Recommendation |
| Notebook 09 | Neural Collaborative Filtering Model |
| Notebook 10 | Movie Recommendation using NCF |
| Notebook 11 | Model Evaluation |
| Notebook 12 | Performance Comparison |
| Notebook 13 | Streamlit Web Application |
| Notebook 14 | Project Conclusion & Future Work |

---

# 🧠 Neural Collaborative Filtering (NCF) Architecture

The recommendation model uses **embedding layers** for users and movies, followed by fully connected neural network layers.

```text
User ID
    │
    ▼
User Embedding (50)

                    ───────────────► Concatenation ◄───────────────

Movie Embedding (50)
    ▲
    │
Movie ID

                │
                ▼
          Fully Connected (128)

                │
                ▼
           ReLU Activation

                │
                ▼
            Dropout (0.2)

                │
                ▼
          Fully Connected (64)

                │
                ▼
           Output Layer

                │
                ▼
        Predicted Movie Rating
```

---

# 📊 Model Performance

## Baseline Model

| Metric | Score |
|---------|------:|
| RMSE | **1.1239** |
| MAE | **0.9420** |

---

## Neural Collaborative Filtering

| Metric | Score |
|---------|------:|
| RMSE | **0.9163** |

---

# 📈 Performance Comparison

| Model | RMSE |
|--------|-----:|
| Baseline | 1.1239 |
| Neural Collaborative Filtering | **0.9163** |

The Neural Collaborative Filtering model significantly reduced the prediction error compared to the baseline model, demonstrating the effectiveness of deep learning for personalized recommendation.

---

# 📷 Screenshots

The following screenshots demonstrate the working application.

---

### 🏠 Home Page

![Home Page](images/home_page.jpg)

---

### 🎬 Recommendation Results (Part 1)

![Recommendation Results](images/recommendations_1.jpg)

---

### 🎬 Recommendation Results (Part 2)

![Recommendation Results](images/recommendations_2.jpg)
### 🧠 Model Workflow

```
images/workflow.png
```

---

### 🏗 Architecture Diagram

```
images/architecture.png
```

---
# ⚙️ Installation Guide

Follow the steps below to run this project on your local machine.

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/SarveshKP-NitD/sarvesh_kr_Recommendation-System.git
```

---

## 2️⃣ Navigate to the Project Directory

```bash
cd sarvesh_kr_Recommendation-System
```

---

## 3️⃣ Create a Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Streamlit Application

Start the Streamlit application by executing:

```bash
streamlit run app.py
```

After a few seconds, Streamlit will automatically open in your web browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

---

# 🚀 How to Use

### Step 1

Open the Streamlit application.

---

### Step 2

Select a **User ID** from the sidebar.

Example:

```
User ID = 1
```

---

### Step 3

Choose the number of recommendations.

Example:

```
Top N = 10
```

---

### Step 4

Click

```
🎬 Recommend Movies
```

---

### Step 5

The application predicts ratings for all unseen movies using the trained Neural Collaborative Filtering model and displays the highest-rated recommendations.

---

# 📋 Sample Recommendation Output

| Rank | Movie | Predicted Rating |
|------|-------------------------------|----------------:|
| 1 | Casablanca (1942) | 4.80 |
| 2 | Close Shave, A (1995) | 4.79 |
| 3 | Pather Panchali (1955) | 4.63 |
| 4 | Titanic (1997) | 4.58 |
| 5 | Fille seule, La (1995) | 4.58 |
| 6 | Shadowlands (1993) | 4.51 |
| 7 | Thin Man, The (1934) | 4.50 |
| 8 | Raise the Red Lantern (1991) | 4.49 |
| 9 | Christmas Carol, A (1938) | 4.49 |
| 10 | As Good As It Gets (1997) | 4.46 |

---

# 📊 Project Highlights

✅ End-to-End Recommendation System

✅ MovieLens 100K Dataset

✅ Neural Collaborative Filtering (PyTorch)

✅ User & Movie Embedding Learning

✅ FAISS Vector Similarity Search

✅ Personalized Movie Recommendations

✅ Streamlit Web Application

✅ RMSE Evaluation

✅ GitHub Portfolio Project

---

# 🎯 Skills Demonstrated

This project demonstrates practical experience with:

- Recommendation Systems
- Deep Learning
- Neural Collaborative Filtering
- Embedding Learning
- Vector Similarity Search (FAISS)
- Data Preprocessing
- Exploratory Data Analysis
- Model Training
- Model Evaluation
- Streamlit Deployment
- Git & GitHub

---

# 💡 Key Learnings

Through this project, the following concepts were explored and implemented:

- Collaborative Filtering techniques
- Neural recommendation models
- User and movie embedding generation
- Recommendation ranking
- Similarity search using FAISS
- Deep learning model training with PyTorch
- End-to-end ML project development
- Interactive deployment using Streamlit

---
# ⚠️ Limitations

Although the project achieves strong recommendation performance, there are some limitations:

- Recommendations are based only on historical ratings.
- New users (Cold Start Problem) have limited recommendations.
- Movie metadata such as genres, actors, and directors are not incorporated.
- Temporal changes in user preferences are not modeled.
- Movie posters and trailers are not included in the current Streamlit application.

---

# 🚀 Future Improvements

The following enhancements can further improve the system:

- 🎬 Integrate TMDB API to display movie posters and descriptions.
- ❤️ Add a "Favorite Movies" feature for users.
- ⭐ Allow users to submit new ratings directly from the application.
- 🤖 Implement Transformer-based recommendation models.
- 📱 Deploy the application on Streamlit Cloud or Hugging Face Spaces.
- ☁️ Store embeddings in a cloud database for scalability.
- 🔍 Hybrid Recommendation System using both collaborative and content-based filtering.
- 📊 Real-time recommendation updates.
- 🌐 User authentication and profile management.
- 🎥 Movie trailer integration using YouTube API.

---

# 📝 Conclusion

This project demonstrates the complete development lifecycle of a modern recommendation system using Deep Learning.

Starting from the MovieLens 100K dataset, the project covers:

- Data preprocessing
- Exploratory Data Analysis
- User and movie encoding
- Embedding learning
- Neural Collaborative Filtering (NCF)
- FAISS vector similarity search
- Model evaluation
- Interactive Streamlit deployment

The Neural Collaborative Filtering model achieved an RMSE of **0.9163**, outperforming the baseline recommendation model (RMSE = **1.1239**).

This project highlights practical skills in machine learning, deep learning, recommendation systems, and model deployment.

---

# 👨‍💻 Author

**Sarvesh Kumar Pal**

M.Tech – Computer Science & Engineering

National Institute of Technology Delhi

📧 Email: *(Add your email here)*

🔗 GitHub: https://github.com/SarveshKP-NitD

🔗 LinkedIn: *(Add your LinkedIn profile here)*

---

# 🙏 Acknowledgements

This project was developed using the following open-source resources:

- MovieLens 100K Dataset (GroupLens Research)
- PyTorch
- Streamlit
- FAISS
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

Special thanks to the open-source community for providing the tools and datasets that made this project possible.

---

# 📄 License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this project for educational and research purposes.

---

# ⭐ Support

If you found this project helpful:

⭐ Star this repository.

🍴 Fork the repository.

🐛 Report issues.

💡 Suggest improvements.

Contributions are always welcome!

---

# 📬 Contact

For questions, suggestions, or collaboration opportunities, feel free to connect.

**GitHub**

https://github.com/SarveshKP-NitD

---

## ⭐ If you like this project, don't forget to star the repository!

Thank you for visiting this project.
Happy Coding! 