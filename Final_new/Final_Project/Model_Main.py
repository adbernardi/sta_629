#################################################
# here will be the main code for loading in, 
# feature engineering, and eventually training the model 
# for later transfer learning on our determined secondary task 
################################################# 

import pandas as pd 
import numpy as np 

# where our data will be loaded when it is downloaded 

chunksize = 200000 

# converting to feather format for faster loading 
#with pd.read_csv(file_path , chunksize = chunksize) as reader:
#    for i, chunk in enumerate(reader):
#        chunk.to_feather(f'chunk_{i}.feather') 

# this should work... then we can just get to training the model after this 
# we can also do some feature engineering here as well depending on the amex 
# data 

# we can use the same denoising before converting to another feather file 
import warnings 
warnings.simplefilter('ignore') 
from tqdm import tqdm 

# let's just chunk the data before denoising and feathering 
# computation is too large to do all at once 

# loading in the data 
# and then we only want a chunk...
'''
chunk = pd.read_csv('/home/adbucks/Documents/sta_629/Final_Project/train_data.csv', nrows = chunksize) 
print(chunk.head())
print(chunk.shape) 
print(chunk.columns)

# now we can define the denoising function 

def denoise(df):
    df['D_63'] = df['D_63'].apply(lambda t: {'CR': 0, 'XZ': 1, 'XM': 2, 'CO': 3, 'CL': 4, 'XL':5}[t]).astype(np.int8)
    df['D_64'] = df['D_64'].apply(lambda t: {np.nan:-1, 'O':0, '-1':1, 'R':2, 'U':3}[t]).astype(np.int8)

    for col in tqdm(df.columns):
        if col not in ['customer_ID', 'S_2', 'D_63', 'D_64']:
            df[col] = np.floor(df[col]*100) 
    return df 

# now can apply this to the train and test data 

chunk_test = pd.read_csv('/home/adbucks/Documents/sta_629/Final_Project/test_data.csv', nrows = chunksize)
print(chunk_test.head()) 
print(chunk_test.shape) 
print(chunk_test.columns) 

# now we can apply the denoising function to the training and testing data 

chunk = denoise(chunk) 
chunk.to_feather('/home/adbucks/Documents/sta_629/Final_Project/train.feather') 

chunk_test = denoise(chunk_test) 
chunk_test.to_feather('/home/adbucks/Documents/sta_629/Final_Project/test.feather')
''' 

# now just have to do this same thing for the labels 
# chunking the labels 
train_labels = pd.read_csv('/home/adbucks/Documents/sta_629/Final_Project/train_labels.csv', 
                           nrows = chunksize)

print(train_labels.head()) 
print(train_labels.shape) 
print(train_labels.columns) 

# now can make a feather function and integrate with the rest of the workflow 
labels = train_labels.to_feather('/home/adbucks/Documents/sta_629/Final_Project/train_labels.feather') 


