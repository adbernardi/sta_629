###################################################################

# We'll just treat this as a sandbox for Homework 2, and we can 
# put things more formally momentarily. 

###################################################################

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

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
knn_5 = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train) # we can try 5 as well 
knn_9 = KNeighborsClassifier(n_neighbors = 9).fit(X_train, y_train) # and 9

# we can now predict 
y_knn_pred = knn.predict(X_test) 
y_knn_pred_5 = knn_5.predict(X_test)
y_knn_pred_9 = knn_9.predict(X_test)
print(y_knn_pred) 
print(y_knn_pred_5)
print(y_knn_pred_9)

# can try and get the score here as well 
knn_accuracy = knn.score(X_test, y_test)
print(knn_accuracy)
knn_accuracy_5 = knn_5.score(X_test, y_test) 
print(knn_accuracy_5) 
knn_accuracy_9 = knn_9.score(X_test, y_test) 
print(knn_accuracy_9) 


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
from sklearn.linear_model import LogisticRegression 

log_reg = LogisticRegression(random_state = 42).fit(X_train, y_train) # default for other arguments 

# now we can predict 
y_log_pred = log_reg.predict(X_test)

# and get the accuracy 
log_accuracy = log_reg.score(X_test, y_test) 
print(log_accuracy) 

# and the confusion matrix 
log_confusion = confusion_matrix(y_test, y_log_pred) 
print(log_confusion) 

# we will now try regularized logsitic regression with an assigned penalty, which we can decide 
# we can try the functions default penalty and then adjust to elastic net after that 
log_reg_l1 = LogisticRegression(penalty = 'l1', solver = 'saga', random_state = 42).fit(X_train, y_train) # adjust solver to deal with l1 penalty

y_log_l1_pred = log_reg_l1.predict(X_test)

log_l1_accuracy = log_reg_l1.score(X_test, y_test) 
print(log_l1_accuracy) 

# just confirming the confusion matrix is the same 
log_l1_confusion = confusion_matrix(y_test, y_log_l1_pred)
print(log_l1_confusion)

# we can now try the elastic net penalty 
log_reg_en = LogisticRegression(penalty = "elasticnet", solver = 'saga', l1_ratio = 0.5, random_state = 42).fit(X_train, y_train) # matching solver with elastic net penalty 

y_log_en_pred = log_reg_en.predict(X_test) 

log_en_accuracy = log_reg_en.score(X_test, y_test) 
print(log_en_accuracy)

# confusion matrix 
log_en_confusion = confusion_matrix(y_test , y_log_en_pred)
print(log_en_confusion)

# still getting good to solid performance, so now we can try linear SVM 
# think we can implement the linear SVC with this code 
from sklearn.svm import LinearSVC 

lin_svm = LinearSVC(random_state = 42).fit(X_train, y_train) # default arguments everywhere else, penalty and hinge loss 

y_lin_svm_pred = lin_svm.predict(X_test) 

lin_svm_accuracy = lin_svm.score(X_test, y_test) 
print(lin_svm_accuracy)

# confusion matrix 
lin_svm_confusion = confusion_matrix(y_test , y_lin_svm_pred)
print(lin_svm_confusion)

# we can now try the kernel SVM, and maybe even compare kernels 
# we'll start with the radial basis function kernel 
from sklearn.svm import SVC 

rbf_svm = SVC(random_state = 42).fit(X_train, y_train) # default kernel is rbf  

y_rbf_svm_pred = rbf_svm.predict(X_test) 

rbf_svm_accuracy = rbf_svm.score(X_test, y_test) 
print(rbf_svm_accuracy) 

# confusion matrix 

rbf_svm_confusion = confusion_matrix(y_test, y_rbf_svm_pred) 
print(rbf_svm_confusion) 

# we can now try the polynomial kernel 

poly_svm = SVC(kernel = "poly", random_state = 42).fit(X_train, y_train) 

y_poly_svm_pred = poly_svm.predict(X_test) 

poly_svm_accuracy = poly_svm.score(X_test, y_test) 
print(poly_svm_accuracy) 

# confusion matrix 
poly_svm_confusion = confusion_matrix(y_test , y_poly_svm_pred)
print(poly_svm_confusion) 



################################################################### 
# problem 2 code will go here 
################################################################### 

# we will now use the entierty of the training and testing data to compare the models 
# and we will compare OVA and OVO for multi-class SVMs
# we will use both of the whole columns for our training and testing data 

# we'll start with the OVA approach for all of the data 
# preparing the data first 
X_train_w = df_whole.iloc[:, 1:]
y_train_w = df_whole.iloc[: , 0]
X_test_w = df_te_whole.iloc[:, 1:]
y_test_w = df_te_whole.iloc[:,0]

# now the OVA, with all other default arguments 
ova_svm = SVC(decision_function_shape = 'ovr', random_state = 42).fit(X_train_w, y_train_w)

y_ova_pred = ova_svm.predict(X_test_w) 

ova_accuracy = ova_svm.score(X_test_w, y_test_w) 
print(ova_accuracy) 

# confusion matrix 
ova_confusion = confusion_matrix(y_test_w, y_ova_pred) 
print(ova_confusion) 

# now we can try the OVO for the whole data 
ovo_sm = SVC(decision_function_shape = 'ovo', random_state = 42).fit(X_train_w, y_train_w) 

y_ovo_pred = ovo_sm.predict(X_test_w)

ovo_accuracy = ovo_sm.score(X_test_w, y_test_w)
print(ovo_accuracy) 

# confusion matrix 
ovo_confusion = confusion_matrix(y_test_w, y_ovo_pred) 
print(ovo_confusion) 

# we'll try another way for the OVO as I don't think the accuracy and confusion matrix should be the same as OVA 
from sklearn.multiclass import OneVsOneClassifier 

ovo_sm2 = OneVsOneClassifier(SVC(random_state = 42))

ovo_sm2.fit(X_train_w, y_train_w)

y_ovo_pred2 = ovo_sm2.predict(X_test_w) 

ovo_accuracy2 = ovo_sm2.score(X_test_w, y_test_w) 
print(ovo_accuracy2) 

# confusion matrix 
ovo_confusion2 = confusion_matrix(y_test_w, y_ovo_pred2) 
print(ovo_confusion2) 

# we can return the column sums excluding the diagonal for the confusion matrices to see which digits are the most mis-classified 
# we can use numpy for this 
ova_confusion_no_diag = ova_confusion.copy()
np.fill_diagonal(ova_confusion_no_diag, 0) 
col_sums = np.sum(ova_confusion_no_diag, axis = 0)
print("Column Sums for OVA: ", col_sums) 

# and we can do the same for the OVO confusion matrix 
ovo_confusion_no_diag = ovo_confusion2.copy()
np.fill_diagonal(ovo_confusion_no_diag, 0)
col_sums_ovo = np.sum(ovo_confusion_no_diag, axis = 0)
print("Column Sums for OVO: ", col_sums_ovo)
# for the write-up, we can print these accuracies and confusion matrices in a more appealing and presentable way 
# we can also compare the models in a more formal way 

print("Naive Bayes Accuracy: ", round(nb_accuracy, 4)) 
print("K Nearest Neighbors Accuracy: ", round(knn_accuracy, 4)) 
print("Linear Discriminant Analysis Accuracy: ", round(lda_accuracy, 4)) 
print("Logistic Regression Accuracy: ", round(log_accuracy, 4)) 
print("Regularized Logistic Regression (L1 Penalty) Accuracy: ", round(log_l1_accuracy, 4)) 
print("Regularized Logistic Regression (Elastic Net Penalty) Accuracy: ", round(log_en_accuracy, 4)) 
print("Linear SVM Accuracy: ", round(lin_svm_accuracy, 4)) 
print("RBF Kernel SVM Accuracy: ", round(rbf_svm_accuracy, 4)) 
print("Polynomial Kernel SVM Accuracy: ", round(poly_svm_accuracy, 4)) 
print("OVA Multi-Class SVM Accuracy: ", round(ova_accuracy, 4)) 
print("OVO Multi-Class SVM Accuracy: ", round(ovo_accuracy2, 4)) 

































