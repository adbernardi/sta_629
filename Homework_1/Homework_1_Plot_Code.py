# we'll use this code just for the plotting 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model 

X = pd.read_csv("/home/adbucks/Documents/sta_629/Homework_1/X_SRBCT.csv", header = None)
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



