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

knn_gs = GridSearchCV(KNeighborsClassifier(), param_grid, 
                     cv = 5)

knn_grid = knn_gs.fit(X_train, y_train)
print("Finding the best KNN model...")
print(knn_grid.best_params_)
print(knn_grid.best_score_) 

print("The best KNN model has the following parameters: ", knn_grid.best_params_) 
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
from sklearn.svm import LinearSVC  

# establishing our parameter grid as before 
'''
lin_svc_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                      'penalty': ['l1', 'l2'], 
                      'loss': ['hinge', 'squared_hinge']}

lin_svc_gs = GridSearchCV(LinearSVC(), lin_svc_param_grid, 
                          cv = 5)

lin_svc_grid = lin_svc_gs.fit(X_train, y_train) 

print("Finding the best Linear SVM model...") 
print(lin_svc_grid.best_params_) 
print(lin_svc_grid.best_score_) 

print("The best Linear SVM model has the following parameters: ", lin_svc_grid.best_params_)
print("The best Linear SVM model has a train accuracy of: ", lin_svc_grid.best_score_) 
'''
# we can now see that the best linear SVC has a C value of 10 and a gamma value of 0.01.
# we will make predictions with this best model now 
#lin_svc_best = LinearSVC(C = 0.01, loss = 'squared_hinge', penalty = 'l2').fit(X_train, y_train) 
#y_pred_lin_svc = lin_svc_best.predict(X_test)

# evaluating the best model now 
#print(confusion_matrix(y_test, y_pred_lin_svc))
#print(classification_report(y_test, y_pred_lin_svc))

# we see a 93% test accuracy for this linear SVC model, which is good, and right in line with the other models 

# we see that this model also does quite well, with a test accuracy of 95%. 
# we will now use the cross validation for the kernel SVM model, and we will find the best parameters for this 
from sklearn.svm import SVC 

# establishing our parameter grid
'''
svc_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

svc_gs = GridSearchCV(SVC(), svc_param_grid, 
                      cv = 5)

svc_grid = svc_gs.fit(X_train, y_train)

print("Finding the best Kernel SVM model...") 
print(svc_grid.best_params_) 
print(svc_grid.best_score_) 
'''
# we have a C value of 10 and a polynomial kernel with an impressive 98% train accuracy, so we will now fit the test data on this best model 
svc_best = SVC(C = 10, kernel = 'poly').fit(X_train, y_train)

y_pred_svc = svc_best.predict(X_test)

# evaluating on the test data now 
print(confusion_matrix(y_test, y_pred_svc)) 
print(classification_report(y_test, y_pred_svc)) 

# we see that this model also does quite well, with a test accuracy f1 score of 95%.
# we can include this information in the write up and then move on to problem 2. 

############################################################################################################
# Problem 2 
# Write a function to perform 5-fold cross-validation for the tuning parameter associated with regularized logistic regression 
# compare and contrast results for minimizing the CV error, and employing the one standard error rule 
# associated with the following loss functions 

# 1. Missclassification error 
# 2. Binomial deviance loss 
# 3. Hinge loss 

# how do the selected tuning parameters perform in terms of test error? 

############################################################################################################ 

# how can we select a model within one standard error of the minimum cross-validation error?
# we will introduce this for our regularized logistic regression model, and we will compare the two methods 
# for the three loss functions. 

# we will write a function for 5-fold cross-validation for the tuning parameter associated with regularized logistic regression, 
# which in this case we can 

# we can get the standard error by importing this from scipy 
from scipy.stats import sem 

# we can start to tune our regularized logistic regression here 
# now we can try and call this function and see what we get, just using default loss to make sure it works
from sklearn.model_selection import cross_val_score 

# what are we actually trying to do?
# we are trying to find the best C value for the logistic regression model, and we are using 5-fold cross validation to do so 
# for the C value, we are either picking the one that minimizes the loss function for the three loss functions, or we are picking 
# the largest one within one standard error of the minimum cross-validation error. 
# we are applying this general process for three loss functions, and we are comparing the results. 

# we can try and get the cross-validation error for the logistic regression model now, just using GridSearchCV to get the minimum lambda, 
# whatever that hyperparameter is 
'''
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

lr_gs = GridSearchCV(LogisticRegression(), param_grid = param_grid_lr) 

lr_grid = lr_gs.fit(X_train, y_train) # fitting the model 

# now we can get the cross-validation error for the logistic regression model, 
# and return our parameters for our best performing model in terms of cross validation error 
# we can think about misclassification rate as 1-accuracy, and we can think about hinge loss as the hinge loss function 

# we'll get our best parameters and then the lowest cross validation error in terms of misclassification rate next 
print(lr_grid.best_params_) 
print(1 - lr_grid.best_score_) 
'''
# we see that our best parameter is a C value of 0.1, and the lowest cross-validation error in terms of mis-classification rate is 0.05. 

# now we can repeat this process in terms of hinge loss 

# now getting best parameters and the lowest cross-validation error in terms of hinge loss 
# trying to implement hinge loss in another way 
# we'll stick with our best model from before, and we'll try to get the hinge loss 
'''

from sklearn.metrics import hinge_loss 

lr_best_hinge = LogisticRegression(C = 0.1).fit(X_train, y_train) 

y_pred_hinge = lr_best_hinge.predict(X_test) 

print(y_pred_hinge.shape) 

# will have to re-shape for the hinge loss with labels 
# including the labels 
labels = np.unique(y_test)

# we will have to change the shape of the predictions to include the labels, so we want (1009, 6) rather htan the current (1009,)  
y_pred_hinge = np.array([y_pred_hinge == label for label in labels]).T 

# evaluating the hinge loss 
lr_hinge_l = hinge_loss(y_test , y_pred_hinge, labels = labels) 

print(lr_hinge_l) 

'''

# we will probably have to make a user defined function to do cross fold validation with hinge loss 

def cross_val_tune(X, y, loss, param_grid):
    # easier if we import this 
    from sklearn.model_selection import KFold
    from sklearn.metrics import hinge_loss  
    

    kf = KFold() # from the question of the problem itself, keeping defaults otherwise 

    # init the empty list for the cross-validation error 
    cv_errors = []

    # can define the labels for the hinge loss in this problem here 
    labels = np.arange(3, 9) # digits 3 through 8 

    for train_index, test_index in kf.split(X):
        cv_X_train, cv_X_test = X.iloc[train_index], X.iloc[test_index]
        cv_y_train, cv_y_test = y.iloc[train_index], y.iloc[test_index]

        # now doing grid search for the tuning hyperparameter in our case 
        lr_gs = GridSearchCV(LogisticRegression(), param_grid, 
                             cv = 5)  

        # now fitting the model 
        lr_grid = lr_gs.fit(cv_X_train, cv_y_train) # adjusting for the CV function  

        # now need to find a way to assess the cross-validation error, using hinge as the loss function 
        if loss == 'hinge_loss':
            # implementing multi-class hinge loss 
            y_pred_hinge = lr_grid.predict(cv_X_test) 

            # including the labels and getting the actual hinge loss 
            lr_hinge_loss = hinge_loss(cv_y_test , y_pred_hinge , labels = labels) # getting the hinge loss 

        # keeping track of the errors 
        cv_errors.append(lr_hinge_loss) # keeping track of our hinge losses for one standard error calculation 

    return cv_errors

# now can try calling the function 
# def param grid for LogisticRegression 

#cross_val_tune(X_train, y_train, 'hinge_loss', param_grid_lr)

# trying another version of multi-class hinge loss below here 
from sklearn.preprocessing import LabelBinarizer 

'''
def multi_class_hinge(y_true, y_pred, labels):
    lb = LabelBinarizer()
    lb.fit(labels) 

    # getting the binarized true values 
    y_true_binarized = lb.transform(y_true) * 2 - 1 # getting the binarized true values 

    # now calculating the hinge loss 
    losses = np.maximum(0, 1 - y_true_binarized * y_pred) 
    loss = np.mean(losses) 

    return loss 
'''
# now we can try and call this function 
labels = np.arange(3, 9) 

# basic model 
#lr_new = LogisticRegression(C = 0.1).fit(X_train, y_train)
#lr_pred = lr_new.predict(X_test)

#hinge_loss_value = multi_class_hinge(y_test, lr_pred, labels)

#print(hinge_loss_value) 

# trying hinge loss agian 
# trying this with a linear SVC to use the hinge loss function 
from sklearn.svm import LinearSVC 

lin_svc = LinearSVC(loss = 'hinge', max_iter = 10000) 

# now establishing hte parameter grid 
param_grid_lin_svc = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

lin_svc_gs = GridSearchCV(lin_svc , param_grid = param_grid_lin_svc, cv = 5) 

# fitting 
lin_svc_grid = lin_svc_gs.fit(X_train, y_train) 

# getting the best parameters 
print("The best parameters using Hinge Loss are: ", lin_svc_grid.best_params_) 
print("The best score using Hinge Loss is: ", lin_svc_grid.best_score_) 

# can dig a little deeper to do the one standard error rule 
mean_test_score = lin_svc_grid.cv_results_['mean_test_score']
std_test_score = lin_svc_grid.cv_results_['std_test_score']

# finding the best to use the standard error rule 
best_score_idx = np.argmax(mean_test_score) 
best_score = mean_test_score[best_score_idx]
best_score_std = std_test_score[best_score_idx]

# getting the threshold now 
thresh = best_score - best_score_std 

# we now want to find the argmax of C values that are within one standard error of the best score, as in hte one standard error rule 
candidate_indices = [i for i, score in enumerate(mean_test_score) if score > thresh]
simplest_idx = candidate_indices[np.argmax(mean_test_score[candidate_indices])] 

# getting the best C value in line with the rule 
best_C = lin_svc_grid.cv_results_['param_C'][simplest_idx]
best_score_1se = mean_test_score[simplest_idx] 

print("The best C value using the one standard error rule is: ", best_C) 
print("The best score using the one standard error rule is: ", best_score_1se) 

# we see that the best parameters using hinge loss are a C value of 0.1, and the best score is 0.95, which is the same as using the other method. 

# we can now try and do this same process with the binomial deviance loss function 
# we can use this as our quick workaround for the bonimal deviance loss 
from sklearn.metrics import log_loss, make_scorer 

lin_ll = LogisticRegression(solver = 'liblinear', max_iter = 10000) 

# now establishing the parameter grid 

param_g_ll = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]} 

# trying to DIY a scoring function 
scorer = make_scorer(log_loss, greater_is_better = False, needs_proba = True)
 
gs_ll = GridSearchCV(lin_ll, param_grid = param_g_ll, cv = 5, scoring = scorer) 

# fitting the model 
gs_ll.fit(X_train, y_train) 

mean_test_scores_ll = -gs_ll.cv_results_['mean_test_score'] # flipping sign due to log loss 
std_test_scores_ll = gs_ll.cv_results_['std_test_score']

# now trying to find the minimum 
best_score_idx_ll = np.argmin(mean_test_scores_ll) 
best_score_ll = mean_test_scores_ll[best_score_idx_ll]
best_score_std_ll = std_test_scores_ll[best_score_idx_ll]

# now finding the threshold 
thresh_ll = best_score_ll + best_score_std_ll 

# finding the simplest model that is within one standard error of the best model 
candidate_indices_ll = [i for i, score in enumerate(mean_test_scores_ll) if score < thresh_ll]
simplest_idx_ll = min(candidate_indices_ll, key = lambda idx: param_g_ll['C'][idx])

# getting the best C value in line with the rule 
best_params_ll_1se = gs_ll.cv_results_['params'][simplest_idx_ll]
best_score_ll_1se = mean_test_scores_ll[simplest_idx_ll]

print("The best parameters using Binomial Deviance Loss are: ", best_params_ll_1se) 
print("The best score using Binomial Deviance Loss is: ", best_score_ll_1se) 






