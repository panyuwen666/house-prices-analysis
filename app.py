import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Comprehensive House Price Analysis", layout="wide")

# Title and intro
st.title("Comprehensive House Price Analysis")
st.write("This app replicates the Kaggle analysis of the House Prices dataset.")
st.markdown("- **Data Source:** [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)")
st.markdown("- **Analysis Based on:** [Comprehensive Data Exploration with Python](https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python)")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    return df

df = load_data()

# Data Overview
st.header("Data Overview")
st.write(f"Dataset shape: {df.shape}")
st.subheader("First 5 rows:")
st.dataframe(df.head())

# Missing Values
st.header("Missing Values Analysis")
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_values = missing_values[missing_values > 0]
st.subheader("Missing Values Count")
st.bar_chart(missing_values)

# SalePrice Distribution
st.header("SalePrice Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, ax=ax, color='skyblue')
ax.set_title("Distribution of SalePrice")
ax.set_xlabel("SalePrice")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Numerical Features Correlation
st.header("Feature Correlation Heatmap")
numerical_df = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numerical_df.corr()
fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', ax=ax, annot=False)
st.pyplot(fig)

# Top Correlated Features with SalePrice
st.header("Top Features Correlated with SalePrice")
top_corr = corr_matrix['SalePrice'].sort_values(ascending=False).head(11)
st.bar_chart(top_corr)

# Numerical Features vs SalePrice
st.header("Numerical Features vs SalePrice")
numerical_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']
for col in numerical_cols:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df[col], y=df['SalePrice'], ax=ax, color='orange')
    ax.set_title(f"{col} vs SalePrice")
    st.pyplot(fig)

# Categorical Features Analysis
st.header("Categorical Features Analysis")
categorical_cols = ['MSZoning', 'LotShape', 'Neighborhood', 'HouseStyle', 'SaleCondition']
for col in categorical_cols:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=df[col], y=df['SalePrice'], ax=ax, palette='Set2')
    ax.set_title(f"{col} vs SalePrice")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Outlier Detection (GrLivArea example)
st.header("Outlier Check for GrLivArea")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'], ax=ax, color='green')
ax.set_title("GrLivArea vs SalePrice (Outlier Check)")
st.pyplot(fig)