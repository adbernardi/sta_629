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
#print(df_whole.head())
print(df_whole.iloc[:,0].unique())

# now slicing the data frame
print(df_whole.iloc[:,0].values)

digits = range(3, 9)

# need to standardize the column names to further prepare the data 
# want to show the first column name as the digit, and the rest as the pixel values 
# we can do this by renaming the columns 
df_whole.columns = ['Digit'] + ['Pixel' + str(i) for i in range(1, 257)]
print(df_whole.head())


df_train = df_whole.loc[df_whole.iloc[:,0].isin(digits)]

print(df_train.head()) 
print(df_train.iloc[:,0].unique())
print(df_train.shape)

# looks good, so we can say that we have our training data of interest

# we can now repeat the process and ensure that we have the right records for the test data 
df_te_whole = pd.read_csv('/home/adbucks/Downloads/zip.test', delim_whitespace = True)

# doing the same checks before filtering in the same way 
'''
print(df_te_whole.head())
print(df_te_whole.iloc[:,0].unique())
print(df_te_whole.iloc[:,0].values)
print(df_te_whole.shape)
'''
# doing the digits in the same way
# we can now re-name the columns in the same way 
df_te_whole.columns = ['Digit'] + ['Pixel' + str(i) for i in range(1, 257)]
print(df_te_whole.head()) 

df_test = df_te_whole.loc[df_te_whole.iloc[:,0].isin(digits)]

print(df_test.head())
print(df_test.iloc[:,0].unique())
print(df_test.shape)

# now we have our training and test data, and we can compare models 
# we'll do some additional implementation for the naive bayes classifier 
import random 
from sklearn.metrics import confusion_matrix 

# we can now start preparing the data for actual modeling and analysis 
# we'll just parse out the training and testing data now 
X_train = df_train.iloc[:, 1:]
y_train = df_train.iloc[:, 0]

# we'll see if this works

print(X_train.head())
print(y_train.head())
print(X_train.shape)
print(y_train.shape)

# now doing this for the test data 
X_test = df_test.iloc[:, 1:]
y_test = df_test.iloc[:, 0]

'''
print(X_test.head())
print(y_test.head()) 
print(X_test.shape) 
print(y_test.shape) 
'''
# something about the feature names not being the same? we can check 
print(X_train.columns)
# we can try to import the gaussian NB and see how it does 
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB().fit(X_train, y_train) 

# can now try and predict 
y_nb_pred = gnb.predict(X_test)
# quick check 
print(y_nb_pred)

# we can now check the accuracy of the model 
nb_accuracy = gnb.score(X_test, y_test) 
print(nb_accuracy)

# for posterity we can also create a confusion matrix 
nb_confusion = confusion_matrix(y_test, y_nb_pred)
print(nb_confusion) 

# we will now think about the models we want to compare, and they are the following
# 1. Naive Bayes 
# 2. KNN 
# 3. LDA 
# 4. Logistic Regression 
# 5. Regularized Logistic Regression 
# 6. Linear SVM 
# 7. Kernel SVM 

# now that we've done Naive Bayes, we can move on to KNN 
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train) 

# we can now predict 
y_knn_pred = knn.predict(X_test) 
print(y_knn_pred) 

# can try and get the score here as well 
knn_accuracy = knn.score(X_test, y_test)
print(knn_accuracy) 

# and the confusion matrix 
knn_confusion = confusion_matrix(y_test, y_knn_pred) 
print(knn_confusion) 

# we can now use LDA with these same data 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

lda = LinearDiscriminantAnalysis().fit(X_train, y_train) # using default solver and parameters 

# now we can try and predict 
y_lda_pred = lda.predict(X_test) 

# and get the accuracy 
lda_accuracy = lda.score(X_test, y_test)
print(lda_accuracy)

# and the confusion matrix 
lda_confusion = confusion_matrix(y_test, y_lda_pred) 
print(lda_confusion)

# now we can try logistic regression 

































