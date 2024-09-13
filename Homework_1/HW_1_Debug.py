'''

This will serve as a sandbox for debugging the code from Homework_1_Code.py.


'''

import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.linear_model import enet_path 
from sklearn.datasets import make_regression 

X, y, true_coef = make_regression(
    n_samles = 100, n_features = 5, n_informative=2, coef=True, random_state=1870
)

print(true_coef)

alphas, estimated_coef, _ = enet_path(X, y, n_alphas = 3)

print(alphas.shape)

print(estimated_coef)
