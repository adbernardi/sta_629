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
print('CART Accuracy: ', accuracy_score(y_test, y_pred)) 
print('CART F1 Score: ', f1_score(y_test, y_pred, average = 'weighted')) 
print('CART Confusion Matrix: ', confusion_matrix(y_test, y_pred)) 

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
print('Adaboost Accuracy: ', accuracy_score(y_val, y_pred_val)) 
print('Adaboost F1 Score: ', f1_score(y_val, y_pred_val, average = 'weighted')) 
print('Adaboost Confusion Matrix: ', confusion_matrix(y_val, y_pred_val)) 

# we can now try the test set for adaboost 
y_pred_adatest = ada_default.predict(X_test) 

# scoring 
print('Adaboost Test Accuracy: ', accuracy_score(y_test, y_pred_adatest)) 
print('Adaboost Test F1 Score: ', f1_score(y_test, y_pred_adatest, average = 'weighted')) 
print('Adaboost Test Confusion Matrix: ', confusion_matrix(y_test, y_pred_adatest)) 

# can try cross-validation in this way with the validation set
'''
from sklearn.model_selection import cross_validate 
from sklearn.metrics import make_scorer 

scores_ada = cross_validate(ada_default, X_train2, y_train2, 
                            cv = 5, 
                            scoring = ['accuracy', 'f1'],
                            return_train_score = True) 

# printing the scores 
#print(scores_ada) # should give the entire dictionary 
'''

# we can also try this in another way via Grid Search 
'''
from sklearn.model_selection import GridSearchCV 

# trying some different variations for the adaboost 
# we can try different n_estimators and learning rates mainly 

params = {'n_estimators': [10, 25, 50, 250, 1000], 
          'learning_rate': [0.01, 0.1, 0.5, 1]}

# creating the grid search instance 
ada_grid = GridSearchCV(ada_default, params, cv = 5, scoring = 'f1_weighted') 

# fitting 
ada_grid.fit(X_train2, y_train2) 

# extracting the best parameters 
# doing this in a precision and recall sense given the multi-class nature of the problem 
y_pred_grid_val = ada_grid.predict(X_val) 

# can just evaluate these best predictions in the normal way 
print('Accuracy: ', accuracy_score(y_val, y_pred_grid_val)) 
print('F1 Score: ', f1_score(y_val, y_pred_grid_val, average = 'weighted')) 
'''

# we'll try gradient boosting now 
from sklearn.ensemble import GradientBoostingClassifier 

# can just stick with the default values for comparison's sake 
# can do this with the validation set as the test set as well 
grad_def_val = GradientBoostingClassifier() 

# fitting 
grad_def_val.fit(X_train2, y_train2) 

# predicting 
y_pred_grad_val = grad_def_val.predict(X_val) 

# now accuracy and scoring 
print('Gradient Boosting Accuracy: ', accuracy_score(y_val, y_pred_grad_val)) 
print('Gradient Boosting F1 Score: ', f1_score(y_val, y_pred_grad_val, average = 'weighted')) 
print('Gradient Boosting Confusion Matrix: ', confusion_matrix(y_val, y_pred_grad_val)) 

# an insane 98% accuracy with the gradient boosting...

# now we can use the test data with gradient boost given this is the method 
# we are going with 

y_pred_grad = grad_def_val.predict(X_test) 

# now accuracy and scoring 
print('Gradient Boosting Test Accuracy: ', accuracy_score(y_test, y_pred_grad)) 
print('Gradient Boosting Test F1 Score: ', f1_score(y_test, y_pred_grad, average = 'weighted')) 
print('Gradient Boosting Test Confusion Matrix: ', confusion_matrix(y_test, y_pred_grad))

# 96% test accuracy, interesting because the validation set was 98%, which is 
# actually optimistic and the accuracy is still very high 

####################################################
# Section 4: Random Forests 
#################################################### 

# now we can implement random forests and see if we can improve 

from sklearn.ensemble import RandomForestClassifier 

# we might want to start with max depth, but we can also try the default 
# we will probably have to prune after getting the default tree 

rf_base = RandomForestClassifier(random_state = 42) 

# fitting 
rf_base.fit(X_train2, y_train2) # still using the validation set 

# RF allows us to see a bit more, so we can print out a few things of 
# interest here  
# now predicting on the validation set 
y_pred_rf_val = rf_base.predict(X_val) 

# now accuracy and scoring, feature importance, and decision path 
print('Base RF Decision Path: ', rf_base.decision_path(X_val))
print('Base RF Feature Importance: ', rf_base.feature_importances_) 
print('Base RF Accuracy: ', accuracy_score(y_val, y_pred_rf_val))
print('Base RF F1 Score: ', f1_score(y_val, y_pred_rf_val, average = 'weighted')) 
print('Base RF Confusion Matrix: ', confusion_matrix(y_val, y_pred_rf_val))
 
# let's now try on our testing data, as we might have to purne a second 
# tree to avoid overfitting 
y_pred_rf = rf_base.predict(X_test) 

# now accuracy and scoring 
print('Base RF Test Accuracy: ', accuracy_score(y_test, y_pred_rf)) 
print('Base RF Test F1 Score: ', f1_score(y_test, y_pred_rf, average = 'weighted')) 
print('Base RF Test Confusion Matrix: ', confusion_matrix(y_test, y_pred_rf))
 
# 98% test accuracy, which is encouraging 

# finally, we can try pruning a bit and see how that goes 
# we can try a naive max depth of 5, although this is something that 
# mght be best for grid search 

rf_p5 = RandomForestClassifier(max_depth = 5, random_state = 42) # naive 

# fitting on the pre-validation training set 
rf_p5.fit(X_train2, y_train2) 

# predicting on the validation set 
y_pred_rf_p5_val = rf_p5.predict(X_val) 

# accuracy and scoring 
print('Pruned RF Accuracy: ', accuracy_score(y_val, y_pred_rf_p5_val)) 
print('Pruned RF F1 Score: ', f1_score(y_val, y_pred_rf_p5_val, average = 'weighted')) 
print('Pruned RF Confusion Matrix: ', confusion_matrix(y_val, y_pred_rf_p5_val)) 

# purned accuracy increases to 99%, now we can try to the test data 
y_pred_test_rfp5 = rf_p5.predict(X_test) 

# accuracy and scoring 
print('Pruned RF Test Accuracy: ', accuracy_score(y_test, y_pred_test_rfp5)) 
print('Pruned RF Test F1 Score: ', f1_score(y_test, y_pred_test_rfp5, average = 'weighted')) 
print('Pruned RF Test Confusion Matrix: ', confusion_matrix(y_test, y_pred_test_rfp5)) 

# pruned tree gives us a 97% accuracy, which is still very good 
# worth comparing other pruning lengths to see if we can get 
# better or comparable accuracy with saving on compute time 


# we can now move on to the write up 






