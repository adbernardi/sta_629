###################################################################

# We'll just treat this as a sandbox for Homework 2, and we can 
# put things more formally momentarily. 

###################################################################

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import math 

# getting the zip code data from the textbook's website 
# we can try to read in our training data, in ASCII text format
# for expedience, we are limiting our consideration to digits between 3 and 8 
# that said, we will read in the individual digits and then will concatenate them 

df_whole = pd.read_csv('/home/adbucks/Downloads/zip.train', delim_whitespace = True)

# we will now filter the data to only include the digits 3 and 8 
print(df_whole.head())
print(df_whole.iloc[:,0].unique())

# now slicing the data frame
print(df_whole.iloc[:,0].values)

digits = range(3, 9)

df_train = df_whole.loc[df_whole.iloc[:,0].isin(digits)]

print(df_train.head()) 
print(df_train.iloc[:,0].unique())
print(df_train.shape)

# looks good, so we can say that we have our training data of interest 

# we will now think about the models we want to compare, and they are the following
# 1. Naive Bayes 
# 2. KNN 
# 3. LDA 
# 4. Logistic Regression 
# 5. Regularized Logistic Regression 
# 6. Linear SVM 
# 7. Kernel SVM 



