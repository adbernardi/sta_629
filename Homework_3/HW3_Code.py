############################################################################################################
# Model code for HW 3 

# this deals iwth the same data sets from Homework 2,
# and we will work with cross-validation for model selection and to evaluate performance. 
# we will start importing the data, recycling the code where appropriate, 
# and then we will get into the cross-validation and tuning. 

############################################################################################################ 

import pandas as pd 
import numpy as np 

df_whole = pd.read_csv('/home/adbucks/Downloads/zip.train', delim_whitespace = True)

print(df_whole.iloc[:,0].unique())

print(df_whole.iloc[:,0].values)

digits = range(3, 9)

df_whole.columns = ['Digit'] + ['Pixel' + str(i) for i in range(1, 257)]
print(df_whole.head())

# getting only digits 3 through 8 
df_train = df_whole.loc[df_whole.iloc[:,0].isin(digits)]

'''
print(df_train.head()) 
print(df_train.iloc[:,0].unique())
print(df_train.shape)
'''

# ensuring that we have the same as before 

# we can now do this with our future testing data 
df_te_whole = pd.read_csv('/home/adbucks/Downloads/zip.test', delim_whitespace = True)

# renaming the columns in the same way 
df_te_whole.columns = ['Digit'] + ['Pixel' + str(i) for i in range(1, 257)]
print(df_te_whole.head()) 

df_test = df_te_whole.loc[df_te_whole.iloc[:,0].isin(digits)]

'''
print(df_test.head())
print(df_test.iloc[:,0].unique())
print(df_test.shape)
'''
# we can now start divying up the data into the response vector and our feuature matrix, 
# and of course with a train and test split 
X_train = df_train.iloc[:, 1:] 
y_train = df_train.iloc[: , 0]

# doing the same for the test now 
X_test = df_test.iloc[:, 1:]
y_test = df_test.iloc[: , 0]

'''
print(X_train.head())
print(y_train.head())
print(X_train.shape)
print(y_train.shape)
print(X_test.head())
print(y_test.head()) 
print(X_test.shape) 
print(y_test.shape) 
'''
# noow we can move on to the cross-validation and tuning
# we will start with the K Nearest Neighbors model, and we will use cross-validation to find the best value of K 


