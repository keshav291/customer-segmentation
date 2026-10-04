
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("Customer Segmentation Project Started!")

# Sample customer data
data = {
    "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8],
    "Age": [20, 25, 35, 40, 22, 45, 30, 50],
    "AnnualIncome": [20, 25, 70, 80, 22, 90, 60, 100],
    "SpendingScore": [80, 75, 40, 30, 85, 20, 50, 15]
}

df = pd.DataFrame(data)

print("\nCustomer Data:")
print(df)

# Select features for clustering
X = df[["AnnualIncome", "SpendingScore"]]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create 3 customer groups
model = KMeans(n_clusters=3, random_state=42, n_init=10)

df["Cluster"] = model.fit_predict(X_scaled)

print("\nCustomer Groups:")
print(df[["CustomerID", "AnnualIncome", "SpendingScore", "Cluster"]])

# Visualize customer groups
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="AnnualIncome",
    y="SpendingScore",
    hue="Cluster",
    palette="viridis",
    s=150
)

plt.title("Customer Segmentation Analysis")
plt.xlabel("Annual Income (in thousands)")
plt.ylabel("Spending Score")
plt.legend(title="Customer Group")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analyze each customer group
print("\nCustomer Group Analysis:")

analysis = df.groupby("Cluster").agg(
    Total_Customers=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("AnnualIncome", "mean"),
    Average_Spending=("SpendingScore", "mean")
).round(2)

print(analysis)

# Save the results
df.to_csv("customer_segments.csv", index=False)

print("\nCustomer segmentation results saved successfully!")