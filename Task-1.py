# Task 1: Exploring and Visualizing the Iris Dataset

# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris Dataset
iris = sns.load_dataset('iris')

# Display Shape of Dataset
print("===== Dataset Shape =====")
print(iris.shape)

# Display Column Names
print("\n===== Column Names =====")
print(iris.columns)

# Display First Five Rows
print("\n===== First Five Rows =====")
print(iris.head())

# Display Dataset Information
print("\n===== Dataset Information =====")
iris.info()

# Display Summary Statistics
print("\n===== Summary Statistics =====")
print(iris.describe())


# Scatter Plot

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=iris,
    x='sepal_length',
    y='sepal_width',
    hue='species',
    s=100
)
plt.title("Scatter Plot: Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

# Histograms for All Numerical Features

iris.hist(figsize=(10, 8))
plt.suptitle("Histograms of Iris Features")
plt.show()

# Histogram with KDE

plt.figure(figsize=(8, 6))
sns.histplot(
    iris['petal_length'],
    kde=True
)
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

# Box Plot for All Features

plt.figure(figsize=(8, 6))
sns.boxplot(data=iris)
plt.title("Box Plot of Iris Features")
plt.show()

# Box Plot by Species

plt.figure(figsize=(8, 6))
sns.boxplot(
    x='species',
    y='petal_length',
    data=iris
)
plt.title("Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length")
plt.show()