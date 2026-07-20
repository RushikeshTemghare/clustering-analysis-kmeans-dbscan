# Clustering Analysis using K-Means and DBSCAN

A comparative analysis of two popular unsupervised machine learning clustering algorithms—**K-Means** and **DBSCAN**—using synthetic datasets generated with Scikit-learn.

![Python](https://img.shields.io/badge/Python-blue)
![Pandas](https://img.shields.io/badge/Pandas-150458)
![NumPy](https://img.shields.io/badge/NumPy-013243)
![Matplotlib](https://img.shields.io/badge/Matplotlib-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-success)
![Clustering](https://img.shields.io/badge/Clustering-KMeans%20%26%20DBSCAN-blueviolet)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📖 Project Overview

This project demonstrates the application and comparison of two widely used unsupervised machine learning clustering algorithms: **K-Means** and **DBSCAN**.

Unlike supervised learning, clustering aims to discover natural groupings within unlabeled data. The notebook explores how centroid-based and density-based clustering techniques behave on datasets with different structures.

Two synthetic datasets are generated using Scikit-learn to evaluate the strengths and limitations of each algorithm:

- A blob-shaped dataset for K-Means clustering.
- A moon-shaped dataset for comparing K-Means and DBSCAN on non-linearly separable data.

The project also demonstrates the **Elbow Method** to determine the optimal number of clusters for K-Means.

---

# 🎯 Problem Statement

Selecting an appropriate clustering algorithm depends on the underlying structure of the data.

The objective of this project is to explore how different unsupervised learning algorithms identify hidden patterns within unlabeled datasets and compare their effectiveness on datasets with both linearly and non-linearly separable clusters.

The project also illustrates how the Elbow Method can be used to determine an appropriate number of clusters before training a K-Means model.

---

# 📊 Dataset

This project does **not** rely on an external CSV dataset.

Instead, two synthetic datasets are generated programmatically using Scikit-learn.

### Dataset 1 — Blob Dataset

The first dataset is generated using:

```python
make_blobs(n_samples=500, centers=3, cluster_std=4, random_state=42)
```

This dataset contains three naturally separated clusters, making it suitable for demonstrating centroid-based clustering using K-Means.

Dataset summary:

- Total observations: **500**
- Number of features: **2**
- Number of clusters: **3**
- Learning Type: **Unsupervised Learning**

---

### Dataset 2 — Moon Dataset

The second dataset is generated using:

```python
make_moons()
```

The moon-shaped dataset contains non-linearly separable observations and is commonly used to demonstrate scenarios where density-based clustering algorithms such as DBSCAN outperform centroid-based approaches.

Dataset summary:

- Synthetic moon-shaped observations
- Two numerical features
- Non-linear cluster structure
- Learning Type: **Unsupervised Learning**

---

# 🛠️ Technologies Used

- 🐍 Python
- 📓 Jupyter Notebook
- 🐼 Pandas
- 🔢 NumPy
- 📊 Matplotlib
- 📈 Seaborn
- 🤖 Scikit-learn
- 📍 K-Means Clustering
- 🌐 DBSCAN
- 📏 StandardScaler
- 💻 Visual Studio Code
- 🌐 Git
- 📂 GitHub

---

# ⚙️ Machine Learning Workflow

This project follows a complete unsupervised machine learning workflow to generate synthetic datasets, preprocess the data, apply clustering algorithms, and compare their performance on different cluster structures.

1. Import Required Libraries
2. Generate Synthetic Datasets
3. Create Pandas DataFrames
4. Explore the Generated Data
5. Standardise Features using StandardScaler
6. Determine the Optimal Number of Clusters using the Elbow Method
7. Train the K-Means Clustering Model
8. Visualise K-Means Clusters
9. Generate a Non-Linear Moon Dataset
10. Apply K-Means Clustering
11. Apply DBSCAN Clustering
12. Compare Clustering Results

---

# 📊 Data Generation

Instead of using a real-world dataset, this project generates synthetic datasets using Scikit-learn.

Two different datasets were created to demonstrate how clustering algorithms behave under different data distributions.

### Blob Dataset

The first dataset was generated using:

```python
make_blobs(
    n_samples=500,
    centers=3,
    cluster_std=4,
    random_state=42
)
```

This dataset contains three naturally separated clusters, making it suitable for centroid-based clustering using the K-Means algorithm.

---

### Moon Dataset

The second dataset was generated using:

```python
make_moons()
```

Unlike the blob dataset, the moon dataset contains non-linearly separable clusters, providing an ideal scenario for comparing K-Means with the density-based DBSCAN algorithm.

---

# 🔍 Data Exploration

After generating each dataset, the observations were converted into Pandas DataFrames for inspection and visualisation.

The generated data was explored to understand the distribution of observations before applying clustering algorithms.

---

# 🧹 Data Preprocessing

Since clustering algorithms rely on distance calculations, feature scaling was performed before model training.

The numerical features were standardised using **StandardScaler**, ensuring that each feature contributed equally during distance-based clustering.

---

# 📈 Determining the Optimal Number of Clusters

Before training the K-Means model on the blob dataset, the **Elbow Method** was applied to determine the most appropriate number of clusters.

The process involved:

- Training multiple K-Means models with different values of **K**
- Calculating the inertia for each model
- Plotting the Elbow Curve
- Identifying the point where increasing the number of clusters produced diminishing improvements

The Elbow Method helped select the final value of **K** used for clustering.

---

# 🤖 Clustering Algorithms Implemented

This project compares two widely used unsupervised machine learning clustering algorithms.

## K-Means Clustering

K-Means is a centroid-based clustering algorithm that partitions observations into a predefined number of clusters by minimising the distance between observations and their assigned cluster centroids.

The K-Means algorithm was applied to:

- The blob dataset
- The moon dataset

The resulting cluster assignments were visualised using scatter plots.

---

## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN is a density-based clustering algorithm that groups observations based on the density of neighbouring points rather than predefined centroids.

Unlike K-Means, DBSCAN:

- Does not require specifying the number of clusters in advance.
- Can identify clusters of arbitrary shapes.
- Detects noise and outlier observations automatically.

DBSCAN was applied to the moon dataset to demonstrate its ability to identify non-linear cluster structures.

---

# 📊 Data Visualisation

Several visualisations were created throughout the project to better understand the clustering process and compare algorithm performance.

The notebook includes:

- Scatter plot of the generated blob dataset
- Elbow Curve for selecting the optimal number of clusters
- Scatter plot showing K-Means clustering results
- Scatter plot of the generated moon dataset
- K-Means clustering visualisation on the moon dataset
- DBSCAN clustering visualisation on the moon dataset

These visualisations provide an intuitive comparison of centroid-based and density-based clustering techniques.

---

# 🔄 Algorithm Comparison

The project compares how different clustering algorithms perform on datasets with varying structures.

### K-Means

- Performs well on compact, spherical clusters.
- Requires the number of clusters to be specified before training.
- Uses centroid distance to assign observations.

### DBSCAN

- Performs well on non-linearly separable datasets.
- Does not require specifying the number of clusters beforehand.
- Identifies clusters based on density.
- Automatically detects noise and outliers.

The comparison demonstrates that while K-Means performs effectively on well-separated blob-shaped clusters, DBSCAN is better suited to datasets containing complex, non-linear cluster structures.

---

# 📊 Results

This project explored the performance of two widely used unsupervised clustering algorithms on synthetic datasets with different underlying structures.

## K-Means Clustering (Blob Dataset)

The K-Means algorithm was first applied to a blob-shaped dataset containing three naturally separated clusters.

Key observations:

- The Elbow Method was successfully used to determine an appropriate number of clusters.
- K-Means accurately identified the natural grouping of the observations.
- The resulting scatter plot showed well-separated clusters with clearly defined centroids.
- The algorithm performed effectively because the dataset contained compact, approximately spherical clusters.

---

## K-Means Clustering (Moon Dataset)

The K-Means algorithm was then applied to the moon-shaped dataset.

Key observations:

- K-Means struggled to correctly separate the non-linear cluster structure.
- Since K-Means partitions data based on centroid distance, it is less effective when clusters have irregular shapes.
- The visualisation highlighted the limitations of centroid-based clustering on complex datasets.

---

## DBSCAN Clustering (Moon Dataset)

DBSCAN was applied to the same moon-shaped dataset for comparison.

Key observations:

- DBSCAN successfully identified the curved cluster structure.
- The algorithm grouped observations based on density rather than centroid distance.
- Unlike K-Means, DBSCAN was able to capture the non-linear shape of the data more effectively.
- The resulting visualisation demonstrated a more natural separation of the two moon-shaped clusters.

---

## 🏆 Key Findings

- ✅ K-Means performed well on compact, well-separated blob-shaped clusters.
- ✅ The Elbow Method provided an effective approach for selecting the number of clusters.
- ✅ K-Means struggled on non-linearly separable data due to its centroid-based approach.
- ✅ DBSCAN successfully clustered the moon-shaped dataset by using density-based clustering.
- ✅ The project demonstrates that selecting an appropriate clustering algorithm depends on the underlying structure of the dataset.
- ✅ Comparing multiple clustering techniques provides valuable insight into the strengths and limitations of unsupervised learning algorithms.

---

# 💡 Skills Demonstrated

Throughout this project, the following machine learning skills were applied:

- Unsupervised Machine Learning
- Clustering Analysis
- K-Means Clustering
- DBSCAN Clustering
- Synthetic Data Generation
- Feature Scaling
- StandardScaler
- Elbow Method
- Cluster Visualisation
- Comparative Algorithm Analysis
- Data Exploration
- Python Programming
- Scikit-learn

---

# 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Understanding the principles of unsupervised machine learning.
- Generating synthetic datasets using Scikit-learn.
- Applying feature scaling before clustering.
- Determining the optimal number of clusters using the Elbow Method.
- Building and evaluating K-Means clustering models.
- Understanding the limitations of centroid-based clustering.
- Applying DBSCAN for density-based clustering.
- Comparing clustering algorithms on datasets with different geometric structures.
- Visualising clustering results using Matplotlib and Seaborn.

---

# 🚀 Future Improvements

Potential enhancements include:

- Evaluate clustering performance using Silhouette Score and Davies-Bouldin Index.
- Explore Hierarchical Clustering and Agglomerative Clustering.
- Compare additional clustering algorithms such as Mean Shift and OPTICS.
- Apply Principal Component Analysis (PCA) before clustering.
- Perform clustering on real-world datasets.
- Build an interactive Streamlit application for clustering visualisation.

---

# 📂 Repository Structure

```text
clustering-analysis-kmeans-dbscan/
│
├── clustering_analysis_kmeans_dbscan.ipynb
├── clustering_analysis_kmeans_dbscan.py
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# 👤 Author

**Rushikesh Temghare**

MSc Data Science & Artificial Intelligence  
Bournemouth University

---

# 📄 License

This project is licensed under the **MIT License**.
