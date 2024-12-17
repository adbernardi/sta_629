

import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 

# load data 
train = pd.read_feather('/home/adbucks/Documents/train.feather') 
test = pd.read_feather('/home/adbucks/Documents/test.feather')
labels = pd.read_feather('/home/adbucks/Documents/train_labels.feather') 

# define target and features 
X = train.drop('id', axis=1) 
y = labels['target'] 



