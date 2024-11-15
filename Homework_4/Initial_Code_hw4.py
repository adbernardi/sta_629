####################################################
# Homework 4 
# This will serve as the initial code for Homework 4, the bagging, tree based methods, and boosting homework 
# also random forests
#################################################### 
# starting the necessary imports 
import pandas as pd 
import numpy as np 


# we'll start by importing the data 
df_train = pd.read_csv('/home/adbucks/Documents/sta_629/Homework_4/authorship_training.csv')

df_test = pd.read_csv('/home/adbucks/Documents/sta_629/Homework_4/authorship_testing.csv') 

# now we can take a peek 

print(df_train.head(10)) 
print(df_test.head(10))
# looks right 

# here, we have 69 columns of words and their counts 
# and the final count is the author name 
# we are going to build models to predict author name based on 
# the word count for the 69 most common words 

####################################################
# in general, we'll use the following methods 
# 1. Classification Tree 
# 2. Bagging 
# 3. Boosting 
# 4. Random Forests 
#################################################### 

# we'll start by building a CART, and this can be a baseline for the other methods  
# given this is a multi-class classification problem with the authors, we 
# will import the tree method accordingly 
from sklearn import tree 

# further preparing the data 
X_train = df_train.iloc[:, :69]
y_train = df_train.iloc[:, 69] 

# testing 
#print(X_train.head(10)) 
#print(y_train.head(10))

# looks good so we can keep going 
# creating the CART instance 
cart = tree.DecisionTreeClassifier() 

# fitting 
cart.fit(X_train, y_train) 

# now predicting and evaluating 
# partitioning first 
X_test = df_test.iloc[:, :69] 
y_test = df_test.iloc[:, 69]

# predicting 
y_pred = cart.predict(X_test) 

#import matplotlib.pyplot as plt 
# we can even plot 
#tree.plot_tree(cart)
#plt.show() 

# a few different scoring metrics 
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix 

# now we can compare 
print('Accuracy: ', accuracy_score(y_test, y_pred)) 
print('F1 Score: ', f1_score(y_test, y_pred, average = 'weighted')) 
print('Confusion Matrix: ', confusion_matrix(y_test, y_pred)) 

# austen, shakespeare, london, and milton are the authors 
# see how many authors we are working with 
#print(df_train.iloc[:, 69].value_counts())

# interesting that we have an 87 percent accuracy for the baseline tree 
# seems like we can move on to bagging now 
# can talk in the write up why we used the F1 score and the confusion matrix over just general mis-classification rate 

#################################################### 
# Section 2: Bagging 
#################################################### 

# we'll see if we can improve our model by bagging, or 
# at least reduce some of the potential overfitting concerns 
# we'll use default arguments for much of this 
# we'll start the importing now 
from sklearn.ensemble import BaggingClassifier 

# creating the instance now 
bag = BaggingClassifier() # using mostly defaults for starters 

bag.fit(X_train, y_train) 

# we'll do predicting now , and then scoring, and we can even try to do 
# parameter extraction, to see if we can drive some inference 
y_pred_bag = bag.predict(X_test)

# now can do our scoring as before 
print('Accuracy: ', accuracy_score(y_test, y_pred_bag)) 
print('F1 Score: ', f1_score(y_test, y_pred_bag, average = 'weighted')) 
print('Confusion Matrix: ', confusion_matrix(y_test, y_pred_bag)) 

# we see the slight jump in accuracy, giving a 92% f1 score 
# we can see the benefit!
# let's try and extract the parameters now

# we can try using out of bag score and see if that makes much difference 
bag_oob = BaggingClassifier(oob_score = True) 
bag_oob.fit(X_train, y_train) 

# now we can extract the oob score 
print('OOB Score: ', bag_oob.oob_score_) 

# worth noting our oob score of 0.95! 
# now we can move on to boosting

#################################################### 
# Section 3: Boosting 
#################################################### 

# we'll now move on to boosting, where we will have to pick 
# between adaboosting and gradient boosting 
# we'll start with adaboosting 

# more parameters with the adaboost, so perhaps we can start with the 
# default values and do some cross-validation with a validation 
# set or something like this 
from sklearn.ensemble import AdaBoostClassifier 

# can try and create a validation set just to 
# keep the test set in a vault 
# here, we'll just use the normal train test split again 
from sklearn.model_selection import train_test_split 

# and can apply this to our existing training data 
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size = 0.25) # using standard validation size 

# now we can train and test on the validation set 
# testing 
print(X_val.head(10)) 
print(X_val.shape) 
print(X_train2.shape)

# can create the instance now 
ada_default = AdaBoostClassifier() 

# fitting on the new training set 
ada_default.fit(X_train2, y_train2) 

# evaluating on the validation set 
y_pred_val = ada_default.predict(X_val) 

# now we can score for a baseline before cross-validation 
print('Accuracy: ', accuracy_score(y_val, y_pred_val)) 
print('F1 Score: ', f1_score(y_val, y_pred_val, average = 'weighted')) 
print('Confusion Matrix: ', confusion_matrix(y_val, y_pred_val)) 


