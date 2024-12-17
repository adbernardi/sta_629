###############################################################
# feature engineering for the amex project 
# will involve cleaning up and working with the feather file 
############################################################### 

import warnings 
warnings.simplefilter('ignore') 

import pandas as pd 
import numpy as np 
import gc, os, random 
import time, datetime 
from tqdm import tqdm 
from sklearn.preprocessing import LabelEncoder

train = pd.read_feather('/home/adbucks/Documents/sta_629/Final_Project/train.feather')
test = pd.read_feather('/home/adbucks/Documents/sta_629/Final_Project/test.feather') 

# let's see what the data looks like 
print(train.head())
print(train.shape) 
print(train.columns) 

print(test.head()) 
print(test.shape) 
print(test.columns) 

# now doing some feature engineering 
def one_hot_encoding(df, cols, is_drop = True):
    for col in cols:
        print('one hot encoding:', col) 
        dummies = pd.get_dummies(pd.Series(df[col]), prefix = 'oneHot_%s'%col)
        df = pd.concat([df, dummies], axis = 1) 
    if is_drop:
        df.drop(cols, axis = 1, inplace = True) 
    return df 

cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", 
                "D_66", "D_68"]
eps = 1e-3  

# now importing the labels for training 
train_y = pd.read_feather('/home/adbucks/Documents/sta_629/Final_Project/train_labels.feather') 
print(train_y.head()) 

train = train.merge(train_y, how='left', on='customer_ID')  
print(train.head()) 
# got it!

# confirming final data eng step before modeling 
train = one_hot_encoding(train, cat_features) 
test = one_hot_encoding(test, cat_features) 

# now we can save the data and move on to our base model 
train.to_feather('/home/adbucks/Documents/sta_629/Final_Project/train_final.feather') 
test.to_feather('/home/adbucks/Documents/sta_629/Final_Project/test_final.feather') 




