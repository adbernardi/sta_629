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

Given that this is a p >> n problem, we will fit and compare the following models; Elastic Net, Lasso, SCAD, and MC+. 

We will start by fitting the model, getting a score, and then we will compare the models based on the score and share some thoughts in greater detail. 

We'll now import the necessary libaries, starting with the Elastic Net. 

'''

# starting with the Elastic Net import 

from sklearn.linear_model import ElasticNet 
from sklearn.datasets import make_regression 

# we will see about a train test split later, but we can just fit the model for now 
# we can try initializing the model and then fitting to the aforementioned data 
model = ElasticNet(random_state = 1870) 
# can now fit in this way 
model.fit(X, y) 

# can try and access the coefficients now 
print(model.coef_)
# and the intercept 
print(model.intercept_)
# I wonder if we can see the length and see how many coefficients we are working with
# we might expect 2307 coefficients, and that is what we get 
#print(len(model.coef_))

# we can get the R^2 score now using the method in sklearn 
print(model.score(X,y))

# we can try to write this more formally 
print("The R^2 score for the Elastic Net model is: ", round(model.score(X,y), 4))

# looks like we have an r-squared of 0.31 or so, and we can see how the other models compare 
# we can also make a regularization path plot for the elastic net model 

from sklearn.linear_model import enet_path, lasso_path  
import matplotlib.pyplot as plt
from itertools import cycle 

# adding some additional parameters 
eps = 5e-3 

# trying the same for the lasso path now 
print("Computing regularization path using the lasso...")
# I think the problem is that we're not flattening y early enough
y = y.flatten()
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps = eps)
print("gut checking...")
print(alphas_lasso.shape, coefs_lasso.shape)
print("Computing the regularization path using the Elastic Net...")
# issue is this is creating a 3d object for alphas and est_coefs, so we need to flatten this 
#alphas, est_coefs, _ = enet_path(X, y, eps = eps) # using the default l1_ratio 
# trying this in a different way
# I think I have to reshape y before doing this 
#y = y.np.reshape(-1,1) 
print("Gut check 1...")
print(X.shape, y.shape, type(y))
# we'll try and convert the y variable to (83,) rather than it's current (83,1)
print("Gut check 2...")
print(X.shape, y.shape, type(y))
alphas_e, est_coefs, _ = enet_path(X, y, eps = eps, n_alphas = 100, l1_ratio = 0.5) # going with defaults in some cases 
print("gut checking...")
print(alphas_e.shape, est_coefs.shape, coefs_lasso.shape)

# NOW we should be able to plot now that the response has been flattened 

# now can try and plot the results with the lasso and enet paths 
# trying the method from the documentation now

# we can try and do the transpose up here 
print("Gut check 3...")
print(type(coefs_lasso), type(est_coefs)) 

# we can try to do the transpose now
# we are having an issue with transposing, so we can try andfix this 
coefs_lasso_t = np.transpose(coefs_lasso)
est_coefs_t = np.transpose(est_coefs) 

print("Gut check 4...")
# the issue is that coefs for the lasso are 3d for some reason, so this needs to change 
print(coefs_lasso_t) 
print(coefs_lasso_t.shape, est_coefs_t.shape)


plt.figure(1) 
colors = cycle(['b', 'r', 'g', 'c', 'k'])
for coef_l, coef_e, c in zip(coefs_lasso, est_coefs, colors):
    # given the dimension I think we have to transpose the est_coefs, we'll try that now
    print('Gut check 3...')
    # checking the data type 
    print(type(coef_l), type(est_coefs))
    
    l1 = plt.semilogx(alphas_lasso, coefs_lasso_t, linestyle = "--", c=c) 
    l2 = plt.semilogx(alphas_e, est_coefs_t, linestyle = "--", c=c) 
plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.title('Lasso and Elastic Net Regularization Paths') 
plt.legend((l1[-1], l2[-1]), ("Lasso", "Elastic Net"), loc = 'lower right')
plt.axis('tight')



#plt.plot(np.log(alphas + eps), est_coefs.T)
'''
colors = ['b', 'r', 'g', 'c', 'k']
# log transforming the alphas 
for coef_e, c in zip(est_coefs, colors):
    l2 = plt.semilogx(alphas, coef_e, linestyle = '--', c = c) 
# have to transform to get the axis right 
# need to trnaspose the est_coefs to get the right shape 
est_coefs = est_coefs.np.transpose() 
l2 = plt.semilogx(alphas, est_coefs, linestyle = "--", c=colors)
plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.title('Elastic Net Regularization Path')
plt.axis('tight') 
plt.show()
'''

# we'll put a pin in the plotting for now, and move on to the Lasso model 
from sklearn import linear_model 

# we can try and fit the model now with a default alpha 
lasso_m = linear_model.Lasso()
# fitting the model now 
lasso_m.fit(X, y)

# we can now get the slopes and intercept now 
print(lasso_m.coef_)
print(lasso_m.intercept_)

# looks like a pretty sparse model in the same way which makes sense for lasso, so we can try and get the score now 
print(lasso_m.score(X, y))
# we can write this more formally using the same format as before 
print("The R^2 score for the Lasso model is: ", round(lasso_m.score(X, y), 4))

# we see this is noticeably worse, and we can move on to the SCAD model now 
# later on, we can make some kind of assessment as to which genes are deemed most effective with relation to the response variable 




                  

















