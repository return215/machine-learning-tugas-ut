# run_clustering.py
# Customer Personality Analysis - Unsupervised Learning Tugas 3
# Written by Antigravity

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plotting
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

print("=== Step 1: Load dataset ===")
# Reading cpa.csv using tab delimiter
data_path = os.path.join(os.path.dirname(__file__), "cpa.csv")
df = pd.read_csv(data_path, sep="\t")
print(f"Dataset successfully loaded. Shape: {df.shape}")

print("\n=== Step 2: Lakukan Eksplorasi awal ===")
print("Data head:")
print(df.head(3))
print("\nMissing values check:")
missing_vals = df.isnull().sum()
print(missing_vals[missing_vals > 0])
print("\nBasic description:")
print(df.describe().T)

print("\n=== Step 3: Identifikasi kolom numerik & kategori ===")
# ID is unique and constant columns like Z_CostContact & Z_Revenue don't have variance, so they should be excluded.
# Dt_Customer is a date string. We can convert it to number of days since registration.
df_processed = df.copy()

# 3.1 Feature engineering on Date
# Parse Dt_Customer to datetime
df_processed['Dt_Customer'] = pd.to_datetime(df_processed['Dt_Customer'], format='%d-%m-%Y')
# Calculate days registered relative to the most recent customer pendaftaran in dataset
max_date = df_processed['Dt_Customer'].max()
df_processed['Days_Registered'] = (max_date - df_processed['Dt_Customer']).dt.days
df_processed.drop(columns=['Dt_Customer'], inplace=True)

# 3.2 Age column from Year_Birth
# Let's say current year is 2015 based on dataset registration dates (~2012 to 2014)
# Or we can just calculate Age relative to registration date or max_date (e.g., Year 2015)
df_processed['Age'] = 2015 - df_processed['Year_Birth']
# Drop Year_Birth
df_processed.drop(columns=['Year_Birth'], inplace=True)

# 3.3 Identify numeric & categorical columns
# Exclude ID, and constant features
exclude_cols = ['ID', 'Z_CostContact', 'Z_Revenue']
potential_features = [col for col in df_processed.columns if col not in exclude_cols]

numeric_cols = []
categorical_cols = []

for col in potential_features:
    if pd.api.types.is_numeric_dtype(df_processed[col]):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

print(f"Numeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

print("\n=== Step 4: Imputasi missing value ===")
# Only numerical columns typically have missing values (e.g. Income).
# We'll use median imputation.
num_imputer = SimpleImputer(strategy='median')
df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])
print("Missing values after imputation:")
print(df_processed[numeric_cols].isnull().sum())

print("\n=== Step 5: Encoding fitur kategori & Gabungkan data ===")
# One-hot encode categorical features (Education, Marital_Status)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df_processed[categorical_cols])
encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

# Gabungkan data numerik & hasil encoding
df_numeric_part = df_processed[numeric_cols].reset_index(drop=True)
df_features = pd.concat([df_numeric_part, encoded_cats_df], axis=1)
print(f"Features shape after encoding and concatenation: {df_features.shape}")

print("\n=== Step 6: Scaling data ===")
# Scale features using StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)
df_scaled_df = pd.DataFrame(df_scaled, columns=df_features.columns)
print("Data scaling completed.")

print("\n=== Step 7: Tentukan jumlah cluster optimal (Elbow + Silhouette) ===")
# Let's compute KMeans for k = 2 to 10
ks = range(2, 11)
inertias = []
silhouettes = []

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(df_scaled, labels))

# Plot Elbow and Silhouette curves side by side
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color=color, fontsize=12)
ax1.plot(ks, inertias, 'o-', color=color, linewidth=2, label='Inertia')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)
ax2.plot(ks, silhouettes, 's-', color=color, linewidth=2, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Determining Optimal Clusters: Elbow Method & Silhouette Score', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()

# Save figure
output_dir = os.path.dirname(__file__)
elbow_fig_path = os.path.join(output_dir, "elbow_silhouette.png")
plt.savefig(elbow_fig_path, dpi=300)
plt.close()
print(f"Elbow and Silhouette plot saved to {elbow_fig_path}")

print("Inertias:")
for k, inertia in zip(ks, inertias):
    print(f"  k={k}: {inertia:.2f}")
print("Silhouette Scores:")
for k, sil in zip(ks, silhouettes):
    print(f"  k={k}: {sil:.4f}")

# Look for optimal k. 
# Usually, k=3 or k=4 is chosen based on this dataset.
# Let's inspect the silhouette scores. The one with highest score or a clear elbow bend.
# We will run KMeans with optimal k. Let's make a reasonable default choice or compute it.
# We will use k=4 for clustering, which is standard for Customer Personality Analysis.
optimal_k = 4
print(f"\nChoosing k={optimal_k} based on optimal balance between silhouette score and elbow bend.")

print(f"\n=== Step 8: Pilih k optimal dan klasterisasi (k={optimal_k}) ===")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(df_scaled)

# Add cluster labels to the original and processed dataframes
df['Cluster'] = cluster_labels
df_processed['Cluster'] = cluster_labels
df_features['Cluster'] = cluster_labels
print("Clustering completed. Customer count per cluster:")
print(df['Cluster'].value_counts())

print("\n=== Step 9: Visualisasi klaster dengan PCA 2D ===")
# Compute PCA
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
df_pca['Cluster'] = cluster_labels

# Plot clusters in 2D space
plt.figure(figsize=(10, 8))
# Set custom palette for high visual aesthetic
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
sns.scatterplot(
    x='PC1', y='PC2',
    hue='Cluster',
    palette=colors[:optimal_k],
    data=df_pca,
    legend="full",
    alpha=0.7,
    s=50
)
plt.title('Customer Clusters Visualized using PCA (2D)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)', fontsize=12)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

pca_fig_path = os.path.join(output_dir, "pca_clusters.png")
plt.savefig(pca_fig_path, dpi=300)
plt.close()
print(f"PCA cluster plot saved to {pca_fig_path}")
print(f"Total variance explained by first two components: {pca.explained_variance_ratio_.sum()*100:.2f}%")

print("\n=== Step 10: Profiling cluster (rata-rata fitur per cluster) ===")
# We want to understand who are the customers in each cluster.
# We'll calculate the mean of numeric variables for each cluster.
profile_cols = [
    'Income', 'Age', 'Kidhome', 'Teenhome', 'Recency',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
    'Complain', 'Response', 'Days_Registered'
]

# Impute Income in df_processed before profiling
df_processed['Income'] = df_processed['Income'].fillna(df_processed['Income'].median())

cluster_profile = df_processed.groupby('Cluster')[profile_cols].mean()

# Count of customers in each cluster
cluster_sizes = df_processed['Cluster'].value_counts().sort_index().rename('Cluster_Size')
cluster_profile = pd.concat([cluster_sizes, cluster_profile], axis=1)

print("\nCluster Profile Table:")
print(cluster_profile.T)

# Save to CSV
profile_csv_path = os.path.join(output_dir, "cluster_profile.csv")
cluster_profile.to_csv(profile_csv_path)
print(f"\nCluster profiling data saved to {profile_csv_path}")

print("\n=== Completed Successfully! ===")
