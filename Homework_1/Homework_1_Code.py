'''

The following code will serve as a submission for the first Homework Assignment. Other questions directly from ESL will likely be written in Latex, but this will handle the 2nd question, 
involved more heavily with coding. 


'''

'''

I. Introduction and Scope of the Problem 

This is a classic, gene expression, n << p problem, where we have a small number of samples, n, and a large number of predictors, p. The goal is to predict the expression of a gene, Y, based on the expression of other genes, X. 

In our case, we have n=83 patients and p=2308 genes, with the response variable being the expression profile of p53, a tumor suppressor gene. 

We want to find which variables and factors are most closely related with p53 expression, and to do this, we will use the Lasso, Ridge, and Elastic Net methods. 

We will start by downloading the data and doing some basic EDA before fitting the models. 


'''

import pandas as pd 
import numpy as np 

# downloading the data
#filepath_1 = 'home/adbucks/Documents/sta_629/Homework_1/X_SRBCT.csv'
#filepath_2 = 'home/adbucks/Documents/sta_629/Homework_1/Y_SRBCT.csv'
X = pd.read_csv("/home/adbucks/Documents/sta_629/Homework_1/X_SRBCT.csv", header = None)
y = pd.read_csv("/home/adbucks/Documents/sta_629/Homework_1/Y_SRBCT.csv", header = None)

# looks okay, so we can convert the response variable to numpy array to be sure 
y = y.to_numpy() 

# doing some additional digging around, we can ensure that none of the predictors are heavily correlated with one another, to prevent colinearity 

print(X.corr()) 

# it seems as if none of these predictors are heavily correlated by printing the matrix, but we can do a quick test to ensure that this is the case 
# we can try and filter the correlation matrix to see if there are any values above 0.5, which would indicate a strong correlation. Other than the obvious 1's on the diagonal, we should not see any values above 0.5. 

#print(X.corr() > 0.5) 
# let's try and get a sum to ensure that the number makes sense 
# let's try testing with a column or two to ensure that the sum is correct 
# let's see what the actual data type is 
corr_matrix = X.corr()

print(corr_matrix.shape)
# maybe we can make this a dataframe to sum and index easier 
corr_matrix = pd.DataFrame(corr_matrix) 
print(corr_matrix.dtypes)

# now we can try slicing through this new dataframe with the first column as a test case 
print((corr_matrix.iloc[:,0] > 0.5).sum())
# looks relatively normal but we can check another column just to be sure 
print((corr_matrix.iloc[:,1] > 0.5).sum())

# that's fine so we can move on, we can check for the VIF after fitting the models just to be sure 

'''

II. Model Fitting and Comparison 

'''


