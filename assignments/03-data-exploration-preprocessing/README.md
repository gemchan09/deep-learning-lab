# 📊 DL-DA02 — Exploratory Data Analysis and Data Preprocessing

A comprehensive analysis exploring exploratory data analysis (EDA), handling missing values, outlier detection, data scaling, and categorical encoding techniques in machine learning workflows.

## Overview

This assignment examines two main parts:

1. **Exploratory Data Analysis (EDA) & Data Preprocessing** — examining data types, calculating descriptive statistics (Mean, Median, Mode), handling missing values, detecting outliers, standardizing features, and encoding categorical variables.
2. **Extended Age Distribution Analysis** — analyzing a larger sample dataset (27 samples) to calculate statistical measures, quartiles, boxplot summaries, and performing data binning/discretization.

## Topics Covered

### 1. Exploratory Data Analysis & Descriptive Statistics

- Identifying numerical and categorical attributes
- Calculating central tendencies: Mean, Median, and Mode
- Handling data anomalies and unreasonable values (e.g., negative salary errors)
- Recognizing skewed distributions (Positive skewness where Mean > Median)

### 2. Data Cleaning & Missing Value Imputation

- Identifying missing values and calculating missing ratios (percentage-based analysis)
- Comparing imputation strategies using Mean versus Median
- Evaluating the robustness of Median imputation in the presence of extreme outliers and skewed features

### 3. Outlier Detection & Feature Scaling

- Detecting outliers using the Interquartile Range ($IQR$) method and upper-bound thresholds ($Q3 + 1.5 \times IQR$)
- Feature scaling using Z-score normalization ($z = \frac{X - \mu}{\sigma}$) to improve model convergence
- Categorical encoding: Binary Encoding and One-Hot Encoding

### 4. Data Reduction & Feature Selection

- Feature selection: Removing irrelevant identifier columns (e.g., ID attributes)
- Evaluating the necessity of dimensionality reduction techniques (e.g., PCA) based on dataset dimensions

## Key Takeaways

- Data cleaning is a critical first step to handle unreasonable entries and missing values before model training.
- Median is more robust than Mean when handling datasets with severe outliers or skewed distributions.
- Z-score normalization removes unit discrepancies and assists gradient-based algorithms in converging faster.
- Feature selection eliminates noise and redundant identifiers, simplifying the dataset structure without losing core information.

## Learning Outcomes

Through this assignment, I explored foundational data preprocessing techniques, practiced handling real-world data imperfections, and understood the mathematical principles behind descriptive statistics and data scaling.

## References

- Sridhar, S. Data Preprocessing and Exploratory Data Analysis Principles.
- scikit-learn documentation. *Preprocessing data*.

## Assignment Information

- **Course:** Deep Learning / Machine Learning Foundations
- **Assignment:** BT02 — Exploratory Data Analysis and Data Preprocessing
- **Repository:** [deep-learning-lab](https://github.com/gemchan09/deep-learning-lab)

---

*This document summarizes my coursework and learning notes. The ideas discussed are based on the assigned exercises and referenced scientific materials.*
