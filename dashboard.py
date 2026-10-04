
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation Dashboard")
st.write("Analyze customer behavior and spending patterns.")

# Load customer data
df = pd.read_csv("customer_segments.csv")

# Summary cards
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Customer Groups", df["Cluster"].nunique())
col3.metric("Average Spending Score", round(df["SpendingScore"].mean(), 2))

st.divider()

# Show customer data
st.subheader("Customer Data")
st.dataframe(df, use_container_width=True)

# Customer group chart
st.subheader("Customer Segmentation")

fig, ax = plt.subplots(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="AnnualIncome",
    y="SpendingScore",
    hue="Cluster",
    palette="viridis",
    s=150,
    ax=ax
)

ax.set_title("Customers by Income and Spending")
ax.set_xlabel("Annual Income (in thousands)")
ax.set_ylabel("Spending Score")

st.pyplot(fig)
plt.close(fig)

# Group analysis
st.subheader("Customer Group Analysis")

summary = df.groupby("Cluster").agg(
    Total_Customers=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("AnnualIncome", "mean"),
    Average_Spending=("SpendingScore", "mean")
).round(2)

st.dataframe(summary, use_container_width=True)

st.success("Customer analysis completed successfully!")