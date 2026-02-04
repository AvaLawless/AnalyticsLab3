# %%
# Step four: Create functions for your two pipelines that produces the train and test datasets. 
# The end result should be a series of functions that can be called to produce the train and test datasets
# for each of your two problems that includes all the data prep steps you took. This is essentially creating
# a DAG for your data prep steps. Imagine you will need to do this for multiple problems in the future so
# creating functions that can be reused is important. You don’t need to create one full pipeline function that
# does everything but rather a series of smaller functions that can be called in sequence to produce the final datasets.
# Use your judgement on how to break up the functions.

# %%
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
# Load college completion data set
college = pd.read_csv('cc_institution_details.csv')
# Load job placement data set
jobs = pd.read_csv('Placement_Data_Full_Class.csv')

# %%
# Filter and clean college completion data set
jobs_filtered = jobs[['degree_t', 'salary']]
print(jobs_filtered.head())
# %%
# Drop missing values
jobs_filtered = jobs_filtered.dropna()
print(jobs_filtered.isnull().sum())
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