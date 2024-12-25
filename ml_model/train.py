import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import joblib

def train_model():
	df = pd.read_csv("C:\\Users\\shashank\\Desktop\\model deployment\\Mall_Customers.csv")
 	# Select features for clustering
	x = df[['Annual Income (k$)', 'Spending Score (1-100)']]
	# Use the Elbow method to find the optimal number of clusters
	wcss = []  # Within-cluster sum of squares
	for i in range(1, 11):
		kmeans = KMeans(n_clusters=i, random_state=42)
		kmeans.fit(x)
		wcss.append(kmeans.inertia_)
	# Plot the Elbow graph
	plt.figure(figsize=(8, 5))
	plt.plot(range(1, 11), wcss, marker='o')
	plt.title('Elbow Method for Optimal Clusters')
	plt.xlabel('Number of Clusters')
	plt.ylabel('WCSS')
	plt.grid(True)
	plt.show()

	# Apply K-means 
	kmeans = KMeans(n_clusters=4, random_state=42)
	df['Cluster'] = kmeans.fit_predict(x)

	# Plot the clusters
	plt.figure(figsize=(8, 6))
	plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=df['Cluster'], cmap='viridis', marker='o')
	plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
	plt.title('Customer Segments')
	plt.xlabel('Annual Income (k$)')
	plt.ylabel('Spending Score (1-100)')
	plt.legend()
	plt.grid(True)
	plt.show()

	joblib.dump(kmeans,"kmeans.pkl")
	return kmeans

if __name__ == "__main__":
 train_model()
	