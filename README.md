# 🏠 King County House Price Prediction

---

## 📌 1. Project Title and General Description

This project involves analyzing a dataset containing one year of house sale prices (from May 2014 to May 2015) in King County, Washington — including Seattle. The main objective is to identify the key factors influencing house prices and build a predictive model using machine learning techniques.

The challenge simulates real-world real estate financial analysis, promoting collaborative problem-solving and practical application of Python skills.

---

## 📊 2. Project Overview

### 🔍 What does this project do?

- Analyzes and models house sale prices using a dataset with 21 housing-related features (e.g., square footage, location, renovation, condition).
- Includes exploratory data analysis (EDA), data cleaning, feature engineering, and development of a regression model.
- Focuses on properties valued at $650,000 and above, providing insights into the high-end real estate market in King County.

### 🎯 What problem does it solve?

- Accurately estimates real estate prices, a critical task for buyers, sellers, investors, and developers.
- Helps guide investment decisions, market research, and pricing strategies.

### 🌍 Potential impact / practical application

- **Real estate agencies:** Refine pricing models and improve client consultations.
- **Investors:** Identify undervalued or overvalued properties.
- **Developers:** Gain data-driven insights into the features that add the most value.
- **Data scientists:** Demonstrate the use of Python and machine learning in real-world financial scenarios.

---

## 📁 3. Dataset Description

- Source: [Kaggle](https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa)
- 21 columns including area metrics, location, ratings, view type, year built and renovated, number of visits, and sale date.
- 21,613 entries.
- Clean data with no missing values or duplicates.

---

## 🎯 4. Research Goal / ML Objective

- Regression problem to predict house prices.
- Model the impact of key real estate features on pricing.

---

## ⚙️ 5. Steps Taken

### 1. Data Cleaning
- Replaced "year built" with house age.
- Simplified "year renovated" to binary variable (renovated or not).
- Treated `id` and `zipcode` as categorical variables.
- Converted `date` column to datetime format.
- Checked multicollinearity with correlation heatmap (no features removed).
- Applied log transformations and capped outliers on skewed continuous variables.

### 2. Exploratory Data Analysis (EDA)
- Used boxplots and distribution plots to identify outliers and distributions.
- Analyzed price distribution, bedrooms, bathrooms, square footage, and location.

### 3. Feature Engineering
- Created spatial clusters based on location.
- Scaled continuous variables.
- Engineered new features like age and renovation indicator.

### 4. Model Training
- Tested Linear Regression, Random Forest, AdaBoost, and XGBoost regressors.
- Evaluated using R² and RMSE metrics.

### 5. Evaluation
- XGBoost had the best initial performance and was selected for further improvements.

---

## 🔍 6. Key Findings

- Before improvements, the most important feature was `grade` (over 40% importance).
- After improvements, feature importance became more balanced:
  - `latitude` and `longitude` became top predictors.
  - `sqft_lot`, `sqft_living`, `age`, and `sqft_living15` gained significant importance.
  - `bathrooms`, `grade`, and `bedrooms` also increased in influence.
  - The newly created cluster feature ranked in the top 15.
- The model became more robust with less reliance on a single feature.
- Performance:
  - Train R²: 0.95
  - Test R²: 0.91

---

## 🧪 7. How to Reproduce the Project

- Python version: 3.11.5
- Main libraries used:
  ```python
  import warnings
  warnings.filterwarnings('ignore')
  import pandas as pd
  import numpy as np
  import seaborn as sns
  import datetime
  import matplotlib.pyplot as plt
  from sklearn.cluster import KMeans
  from xgboost import XGBRegressor
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LinearRegression
  from sklearn.metrics import r2_score, mean_squared_error
  from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
  from sklearn.preprocessing import StandardScaler, RobustScaler
  from sklearn.preprocessing import OneHotEncoder
  from numpy import hstack
 
### 🧾 Files to Run

- `Mini_Project.ipynb`

## 🚀 8. Next Steps / Improvements

- Test improvements applied in XGBoost on other models.
- Evaluate feature importance consistency across models.
- Explore advanced feature engineering and hyperparameter tuning.

## 🗂️ 9. Repository Structure

| File/Folder               | Description                                         |
|---------------------------|-----------------------------------------------------|
| `Mini_Project.ipynb`      | Jupyter notebook with all code and analysis         |
| `Mini_Project.pdf`        | Slide presentation of the business findings         |
| `README.md`               | This file – project overview and documentation      |
