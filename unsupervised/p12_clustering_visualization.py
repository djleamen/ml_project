'''
Clustering and Visualization of Customer Data using K-Means and DBSCAN
From AI and Machine Learning Algorithms and Techniques by Microsoft on Coursera
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

data = {'AnnualIncome': [
        15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5,
        20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5,
        25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5,
        30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5,
        35,   # Normal points
        80, 85, 90  # Outliers
        ],
        'SpendingScore': [
        39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
        69, 72, 75, 78, 81, 84, 87, 90, 93, 96,
        6, 9, 12, 15, 18, 21, 24, 27, 30, 33,
        5, 8, 11, 14, 17, 20, 23, 26, 29, 32,
        56,   # Normal points
        2, 3, 100  # Outliers
]}

df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Normalize the dataset using StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore'])
print(df_scaled.head())

# Initialize and fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_scaled)

# Add cluster labels to the original dataframe
df['KMeans_Cluster'] = kmeans.labels_

# Display the first few rows with cluster labels
print(df.head())

# Plot K-Means clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'],
            c=df['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Initialize and fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan.fit(df_scaled)

# Add DBSCAN cluster labels to the original dataframe
df['DBSCAN_Cluster'] = dbscan.labels_

# Display the first few rows with cluster labels
print(df.head())

# Plot DBSCAN clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'],
            c=df['DBSCAN_Cluster'], cmap='rainbow')
plt.title('DBSCAN Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()
