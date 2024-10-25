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
# we can now start with the grid search for the optimal KNN parameters given the data 
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KNeighborsClassifier 

# establishing the parameter grid 
param_grid = {'n_neighbors': np.arange(1, 25) , 
              'weights': ['uniform', 'distance'], 
              'metric': ['euclidean', 'manhattan']}

'''knn_gs = GridSearchCV(KNeighborsClassifier(), param_grid, 
                     cv = 5)

knn_grid = knn_gs.fit(X_train, y_train)
print("Finding the best KNN model...")
#print(knn_grid.best_params_)
#print(knn_grid.best_score_) 
'''
# seems that we get the best model as having 4 neighbors, euclidean distance, and uniform weights. 
# we will use that model to make predictions now. 

knn_best = KNeighborsClassifier(n_neighbors = 4, metric = 'euclidean', weights = 'uniform').fit(X_train, y_train)

y_pred = knn_best.predict(X_test)

# evaluating our best model now 
from sklearn.metrics import classification_report, confusion_matrix 

print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred)) 

# seems to be our best performer, with 95% test accuracy. 
 
# we will now try regularized logistic regression, in an attempt to find the ideal parameters. 
from sklearn.linear_model import LogisticRegression 

# establishing the parameter grid again 
''''
lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                 'penalty': ['l1', 'l2']} 

lr_gs = GridSearchCV(LogisticRegression(), lr_param_grid, 
                     cv = 5)

lr_grid = lr_gs.fit(X_train, y_train) 
print("Finding the best Logistic Regression model...") 
print(lr_grid.best_params_)
print(lr_grid.best_score_)
'''

# we see that an l2 penalty with a C value of 0.1 is the best model.
# we will now use that model to make predictions. 

lr_best = LogisticRegression(C = 0.1, penalty = 'l2').fit(X_train, y_train) 
y_pred_lr = lr_best.predict(X_test) 

# evaluating our best model now 
print(confusion_matrix(y_test, y_pred_lr)) 
print(classification_report(y_test, y_pred_lr)) 

# quickly, we can also try and do this CV with an elastic net logistic regression model 
'''
lr_param_grid_en = {'C': [0.001, 0.01, 0.1, 1, 10],
                    'l1_ratio': np.arange(0, 1, 0.25),
                    'max_iter': [100]}

lr_gs_en = GridSearchCV(LogisticRegression(penalty = 'elasticnet', solver = 'saga'), lr_param_grid_en, 
                        cv = 5)

lr_grid_en = lr_gs_en.fit(X_train, y_train) 
print("Finding the best Elastic Net Logistic Regression model...") 
print(lr_grid_en.best_params_) 
print(lr_grid_en.best_score_) 
'''

# we see this regularized logistic model performs well, with a test accuracy of 93%. 
# l2 regularization performed best in a cross-validation comparison, but we will likely need to compare with an elastic net model. 

# we will now try cross-validation with a linear SVM model 
from sklearn.svm import SVC 

# establishing our parameter grid as before 
svc_param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

svc_gs = GridSearchCV(SVC(), svc_param_grid, 
                      cv = 5)

svc_grid = svc_gs.fit(X_train, y_train) 

print("Finding the best Linear SVM model...") 
print(svc_grid.best_params_) 
print(svc_grid.best_score_) 

# we can now see that the best linear SVC has a C value of 10 and a gamma value of 0.01.
# we will make predictions with this best model now 
svc_best = SVC(C = 10, gamma = 0.01).fit(X_train, y_train) 
y_pred_svc = svc_best.predict(X_test)

# evaluating the best model now 
print(confusion_matrix(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))

# we see that this model also does quite well, with a test accuracy of 95%. 







