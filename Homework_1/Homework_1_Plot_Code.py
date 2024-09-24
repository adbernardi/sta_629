# we'll use this code just for the plotting 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model 

X = pd.read_csv("/home/adbucks/Documents/sta_629/Homework_1/X_SRBCT.csv", header = None)
# initial check 
y = pd.read_csv("/home/adbucks/Documents/sta_629/Homework_1/Y_SRBCT.csv", header = None)

# looks okay, so we can convert the response variable to numpy array to be sure 
y = y.to_numpy() 

y = y.flatten()

# now the modeling 
from sklearn.linear_model import ElasticNet, Lasso, enet_path, lasso_path 

e_model = ElasticNet(random_state = 1870)

e_model.fit(X, y)

# now the lasso model 
lasso_m = linear_model.Lasso()
lasso_m.fit(X, y)

# now we can try to do the plotting 
# we can try to plot the elastic net path now
eps = 5e-3  
alphas_e, coefs_e, _ = enet_path(X, y, eps = eps) 

# now the lasso path 
alphas_l, coefs_l, _ = lasso_path(X, y, eps = eps)

# now we can try to do a quick print check 
print("Gut check 1...") 
print(alphas_e.shape, coefs_e.shape)
print("Gut check 2...")
print(alphas_l.shape, coefs_l.shape)

# seems like we can try to plot the results now
# have to transpose the coefficients before plotting 
coefs_et = np.transpose(coefs_e)
coefs_lt = np.transpose(coefs_l)
print("Gut check 3...")
print(coefs_et.shape, coefs_lt.shape) 


# seems like the plotting is weird 
# trying another method, just doing elastic net 
plt.plot(alphas_e, coefs_et)
plt.xlabel("alpha") 
plt.ylabel("coefficients") 
ymin, ymax = plt.ylim()
plt.title("Elastic Net Regularization Path") 
plt.axis('tight')
plt.show() 

# this seems to work now! we can go through the lasso path now 
plt.plot(alphas_l, coefs_lt)
plt.xlabel("alpha")
plt.ylabel("coefficients")
ymin, ymax = plt.ylim() 
plt.title("Lasso Regularization Path") 
plt.axis('tight') 
plt.show() 

# now we will move on to the SCAD model 
# SCAD and MC+ model done in R. Now done with the plotting, we will try and 
# get a sense of the most important variables, and try to get a sense of how the models compare 
# we can start with scoring 

print("Elastic Net Score: ", e_model.score(X, y)) # 0.306
print("Lasso Score: ", lasso_m.score(X, y)) # 0.103 

# seems like the elastic net model is better than the lasso model in terms of R squared values 
# we can try and get a sense of the most important variables now 
# I guess we could order the coefficients and see which ones are the most important 

# can probably try and sort the beta values to get a better idea 
# want to try and sort these values to get a better idea of the most important variables 
print(e_model.coef_) # need to try and find a way to sort these 

coefs_e = e_model.coef_ 
coefs_l = lasso_m.coef_

print("Coefficients gut check...")
print(type(coefs_e), type(coefs_l))

# we can try and sort the coefficients now
# want to sort the absolute value of the coefficients 
coefs_e = np.abs(coefs_e) 
coefs_l = np.abs(coefs_l) 

coefs_e = np.sort(coefs_e)[::-1] # reverse order workaround 
coefs_l = np.sort(coefs_l)[::-1] 

print("Elastic Net Coefficients: ", coefs_e) 
print("Lasso Coefficients: ", coefs_l) 

# per the homework, we can pull out the top 10 variables 

top_10_enet = coefs_e[0:10] 
top_10_lasso = coefs_l[0:10] 

print("Top 10 Elastic Net Coefficients: ", top_10_enet) 
print("Top 10 Lasso Coefficients: ", top_10_lasso) 

# now we will just have to see which genes these coefficients correspond to 
# we can try to get the gene names from the original data 
# trying to do this another way 
#coefs_list = np.array(X[coefs_e > 0])
print("Gut check 4...") 
#print(coefs_list.head())
# trying to see if this works 
#print(list(zip(lasso_m.coef_, X.columns)))
# we'll try and get model feature names this way 
#lasso_m.feature_names = list(X.columns)

coefs_l_s = lasso_m.sparse_coef_

print("Gut check 5...") 
print(coefs_l_s) 

# we can try reading in X with the header included and see if that helps 
# trying it another way 
coefs = lasso_m.coef_  

non_zero = [i for i, coef in enumerate(coefs) if coef != 0]

# now trying to find the feature names 
non_zero_feature_names = [X.columns[i] for i in non_zero]
non_zero_values = [coefs[i] for i in non_zero]

# trying to print the results 
for name, value in zip(non_zero_feature_names, non_zero_values):
    print(f'Feature: {name}, Coefficient: {value}')

# I think this makes sense, and we can try to do the same for the elastic net now 
coefs_e = e_model.coef_

non_zero_e = [i for i, coef in enumerate(coefs_e) if coef != 0]

non_zero_feature_names_e = [X.columns[i] for i in non_zero_e]
non_zero_values_e = [coefs_e[i] for i in non_zero_e]

# now trying to fancy print 
for name, value in zip(non_zero_feature_names_e, non_zero_values_e):
    print(f'Feature: {name}, Coefficient: {value}')

# this really helps us get a sense of the most important variables
# we can include these important variables in our write up 











