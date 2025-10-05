import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# -------------------------------
# STEP 1: Create Dummy Customer Data
# -------------------------------
data = {
    "CustomerID": range(1, 21),
    "Age": [25, 34, 45, 23, 52, 46, 56, 44, 36, 29, 48, 60, 32, 40, 28, 50, 42, 37, 31, 27],
    "Annual_Income": [35000, 45000, 62000, 29000, 72000, 61000, 80000, 58000, 52000, 39000,
                      67000, 90000, 43000, 56000, 31000, 75000, 60000, 54000, 42000, 36000],
    "Spending_Score": [39, 81, 6, 77, 40, 80, 17, 76, 94, 73, 14, 99, 82, 79, 40, 15, 13, 90, 70, 65],
    "Total_Purchases": [5, 15, 2, 14, 6, 18, 4, 20, 22, 13, 3, 25, 17, 19, 7, 8, 6, 21, 11, 10]
}

df = pd.DataFrame(data)

# Add total spending (approximation)
df["Total_Spending"] = df["Total_Purchases"] * (df["Annual_Income"] / 1000)

# -------------------------------
# STEP 2: KMeans Clustering
# -------------------------------
X = df[["Age", "Annual_Income", "Spending_Score"]]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

# -------------------------------
# STEP 3: Analysis
# -------------------------------
customer_rev = df.groupby("Cluster").agg({
    "Age": "mean",
    "Annual_Income": "mean",
    "Spending_Score": "mean",
    "Total_Spending": "mean"
}).reset_index()

print("Cluster Summary:")
print(customer_rev)

# -------------------------------
# STEP 4: Visualization + Saving
# -------------------------------

# Chart 1: Distribution of Customers by Cluster
sns.countplot(x="Cluster", data=df, palette="pastel")
plt.title("Customer Distribution by Cluster")
plt.savefig("cluster_distribution.png")   # ✅ Save chart as PNG
plt.show()

# Chart 2: Average Spending per Cluster
sns.barplot(x="Cluster", y="Total_Spending", data=df, palette="viridis")
plt.title("Average Spending by Cluster")
plt.savefig("spending_by_cluster.png")   # ✅ Save chart as PNG
plt.show()

# -------------------------------
# STEP 5: Save Results
# -------------------------------
df.to_csv("customer_segments.csv", index=False)   # Save full data
customer_rev.to_csv("cluster_summary.csv", index=False)   # Save summary

print("✅ Project Completed: Charts & CSV files saved in folder!")
