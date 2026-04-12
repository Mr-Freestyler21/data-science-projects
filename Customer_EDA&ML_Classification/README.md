# Customer Subscription Analysis

## Overview
An end-to-end data science project covering data cleaning, exploratory data analysis (EDA), 
visualisation, and machine learning on a customer subscription dataset of 1000 rows.

## Dataset
The raw dataset contained 11 columns including Age, Gender, Salary, Country, 
Purchase Amount, and Subscription Status. It had multiple data quality issues that 
required cleaning before any analysis could be done.

## Data Cleaning
Issues found and fixed:
- Age stored as text with values like "thirty" and "?" — converted to numeric
- Gender had 6 inconsistent variants — standardised to Male/Female
- Country had abbreviations like NIG, US, UK — mapped to full names
- Salary had currency noise ($, NGN, USD) and "abc" as null placeholder — cleaned to numeric
- Dates had 7 different formats and invalid dates like month 13 — parsed to datetime
- Negative Purchase Amount values — replaced with null
- Subscription Status had typos like "Actve" — corrected
- Malformed emails — nulled out

## Exploratory Data Analysis
Key findings:
- 81% of users are Active subscribers
- United States has the most users (304), Canada the least (109)
- Canada and Nigeria have the highest average purchase amounts
- Male users spend ~$500 more on average than female users
- Inactive users surprisingly spend more on average than Active users
- User signups have declined each year from 2022 to 2024
- Salary and purchase amount have near zero correlation (-0.021)

## Visualisations
- Age, Salary and Purchase Amount distributions
- Users by Gender, Country, Subscription Status and Join Year
- Average Purchase Amount by Country, Gender and Subscription Status

## Machine Learning
Built a Decision Tree Classifier to predict whether a user is Active or Inactive.

- Features used: Salary, Purchase Amount, Gender, Country
- Algorithm: Decision Tree (scikit-learn)
- Train/test split: 80/20
- Accuracy: 69%

The model performed well on Active users (F1-score: 0.80) but struggled with 
Inactive users (F1-score: 0.20) due to class imbalance — 81% of users are Active 
so the model defaulted to predicting Active most of the time. This is a known 
challenge in real-world ML called the class imbalance problem.

The decision tree revealed that Country (specifically United Kingdom) was the 
most important feature for splitting predictions, followed by Purchase Amount and Salary.

## Tools Used
- Python 3.14
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook (VS Code)

## Key Learnings
- Importance of data cleaning before any analysis
- Dropping NaN values locally per chart rather than globally
- One-hot encoding for converting categorical variables to numeric
- Class imbalance and its effect on model performance
- The complete data science workflow: clean → explore → visualise → model