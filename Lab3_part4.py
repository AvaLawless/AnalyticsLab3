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
jobs_filtered = jobs.drop(columns=[
  'ssc_p', 
  'ssc_b', 
  'hsc_p', 
  'hsc_b'])
print(jobs_filtered.head())
# %%
# Drop missing values
jobs_filtered = jobs_filtered.dropna()
print(jobs_filtered.isnull().sum())
# %%
# Apply one-hot encoding to the degree_t column
jobs_encoded = pd.get_dummies(jobs_filtered, columns=[
  'degree_t', 
  'gender', 
  'hsc_s', 
  'workex', 
  'specialisation', 
  'status'
], drop_first=True)
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
college_filtered = college.drop(columns=[
  'basic', 'hbcu', 'flagship', 'nicknames', 'similar', 
  'med_sat_value', 'med_sat_percentile', 'endow_value',
  'endow_percentile', 'vsa_year', 'vsa_grad_after4_first',
  'vsa_grad_after4_first', 'vsa_grad_elsewhere_after4_first',
  'vsa_enroll_after4_first', 'vsa_enroll_elsewhere_after4_first',
  'vsa_grad_after6_first', 'vsa_grad_elsewhere_after6_first',
  'vsa_enroll_after6_first', 'vsa_enroll_elsewhere_after6_first',
  'vsa_grad_after4_transfer', 'vsa_grad_elsewhere_after4_transfer',
  'vsa_enroll_after4_transfer', 'vsa_enroll_elsewhere_after4_transfer',
  'vsa_grad_after6_transfer', 'vsa_grad_elsewhere_after6_transfer',
  'vsa_enroll_after6_transfer', 'vsa_enroll_elsewhere_after6_transfer',
  'city', 'state', 'level', 'site', 'counted_pct', 'chronname'
])
print(college_filtered.head())

# %%
# Drop missing values
college_filtered = college_filtered.dropna()
print(college_filtered.isnull().sum())
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
# Step four: Create functions for your two pipelines that produces the train and test datasets. 

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score

# %%
# Initialize the DecisionTreeClassifier
dtree = DecisionTreeClassifier()
# %%
# Fit the model to the training data (jobs)
dtree.fit(train_jobs.drop('salary', axis=1), train_jobs['salary'])

# %%
# Fit the model to the training data (colleges)
dtree.fit(train_college.drop('student_count', axis=1), train_college['student_count'])

# %%
# Predict on the test data (jobs)
jobs_predictions = dtree.predict(test_jobs.drop('salary', axis=1))

# %%
# Predict on the test data (colleges)
college_predictions = dtree.predict(test_college.drop('student_count', axis=1))

# %%
# Calculate precision score for jobs
jobs_precision = precision_score(test_jobs['salary'], jobs_predictions,
                                 average='weighted', zero_division=0)
print(f'Precision Score for Jobs: {jobs_precision}')

# %%
# Calculate precision score for colleges
college_precision = precision_score(test_college['student_count'], college_predictions,
                                    average='weighted', zero_division=0)
print(f'Precision Score for Colleges: {college_precision}')
