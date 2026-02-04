# Machine Learning Boot Camp - Ava Lawless

# Step one: Review these two datasets and brainstorm problems that could be
# addressed with the dataset. Identify a question for each dataset.
# %%
# Import packages
# %% 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load college completion data set
college = pd.read_csv('cc_institution_details.csv')
print(college.head())


# %%

# Load job placement data set

jobs = pd.read_csv('Placement_Data_Full_Class.csv')
print(jobs.head())
# %%
# Problems that could be addressed with the College Completion dataset: 
# What is the distribution of students attending different types of colleges (private, public, etc)?

# Problems that could be addressed with the Job Placement dataset:
# What is the average salary for graduates based on their degree type?
# What is the proportion of placed versus not placed employees based on degree type?
# What is the average salery based on gender? 

# Step two (College Completion data set): Work through the steps outlined in the examples to include the following elements:

# 1. Write a generic question that this dataset could address.
# How many students attend different types of colleges (private, public, etc)?
# 
# 2. What is an independent Business Metric for your problem? Think about the case study examples we have discussed in class.
# The independent business metric for this problem is the number of students attending each type of college (private, public, etc).
#
# Data preparation:

# %%
# Check data types
print(college.dtypes)

# %%
# Check column names
print(college.columns)

# %%
# Filter data set to only include relevant columns
college_filtered = college[['student_count', 'control']]
print(college_filtered.head())
# %%
# Check for missing values
print(college_filtered.isnull().sum())

# %%
# Check the unique values in the control column
print(college_filtered['control'].unique())
# %%
# Count number of rows that include each type of institution
print(college_filtered['control'].value_counts())

# %%
# Apply one-hot encoding to the control column
college_encoded = pd.get_dummies(college_filtered, columns=['control'], drop_first=True)
print(college_encoded.head())
# %%
# Calculate the prevalence of the target variable
prevalence = college_encoded['student_count'].value_counts(normalize=True)
print(prevalence)
# %%
# Create the necessary data partitions (Train,Tune,Test)
train_college, test_college = train_test_split(college_encoded, test_size=0.2, random_state=42)
print(train_college.shape, test_college.shape)
# %%
# Step three: What do your instincts tell you about the data.
# Can it address your problem, what areas/items are you worried about?
# My instincts tell me that the data can address the problem of understanding the distribution of
# students attending different types of colleges. However, I am worried about the limited number of
# colleges represented in the data set, which may not be representative of the overall population of colleges.

# Step two (Job Placement data set): Work through the steps outlined in the examples to include the following elements:

# 1. Write a generic question that this dataset could address.
# What is the average salary for graduates based on their degree type?

# 2. What is an independent Business Metric for your problem? Think about the case study examples we have discussed in class.
# The independent business metric for this problem is the average salary for graduates based on their degree type

# Data preparation:
# %%
# Check data types
print(jobs.dtypes)

# %%
# Check column names
print(jobs.columns)
# %%
# Filter data set to only include relevant columns
jobs_filtered = jobs[['degree_t', 'salary']]
print(jobs_filtered.head())
# %%
# Check for missing values
print(jobs_filtered.isnull().sum())
# %%
# Drop missing values
jobs_filtered = jobs_filtered.dropna()
print(jobs_filtered.isnull().sum())
# %%
print(jobs_filtered.tail())
# %%
# Check the unique values in the degree_t column
print(jobs_filtered['degree_t'].unique())
# %%
# Count number of rows that include each type of degree
print(jobs_filtered['degree_t'].value_counts())

# %%
# Apply one-hot encoding to the degree_t column
jobs_encoded = pd.get_dummies(jobs_filtered, columns=['degree_t'], drop_first=True)
print(jobs_encoded.head())
# %%
# Calculate the prevalence of the target variable
prevalence = jobs_encoded['salary'].value_counts(normalize=True)
print(prevalence)
# %%
# Create the necessary data partitions (Train,Tune,Test)
train_jobs, test_jobs = train_test_split(jobs_encoded, test_size=0.2, random_state=42)
print(train_jobs.shape, test_jobs.shape)
# %%
# Step three: What do your instincts tell you about the data.
# Can it address your problem, what areas/items are you worried about?
# My instincts tell me that the data can address the problem of understanding the average salary
# for graduates based on their degree type. However, there are only two types of degrees in this data set,
# which are Sci&Tech and Comm&Mgmt (as well as an "Other" category). Additionally, there are significantly
# more Sci&Tech graduates than Comm&Mgmt graduates, which may skew the results.

