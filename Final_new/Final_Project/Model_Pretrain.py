####################################################
# here is where we will include the code 
# for the pretraining of the model 
# using the given labeled data 
#################################################### 

import warnings 
warnings.simplefilter('ignore') 
import pandas as pd 
import numpy as np
import torch 
from torch import nn 
import os 
from torch.utils.data import DataLoader 

# loading the data 
train = pd.read_feather('/home/adbucks/Documents/sta_629/Final_Project/train_final.feather') 
test = pd.read_feather('/home/adbucks/Documents/sta_629/Final_Project/test_final.feather') 

# let's see what the data looks like 
print(train.head()) 
print(train.shape) 
print(train.columns) 

print(test.head()) 
print(test.shape) 
print(test.columns) 

print(train['target'].value_counts()) # looks good!

# now on to the core pytorch code 
# specifying the batch size 
batch_size = 225 # same as before 

# specifying devices for training, if there exists a hardware accelerator
device = (
    "cuda"
    if torch.cuda.is_available() 
    else "mps" 
    if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"Using {device} device")

# now defining the neural net class
# set up the architecture of the NN here as well 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__() 
        self.flatten = nn.Flatten() 
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(225, 24),
            nn.ReLU(), 
            nn.Linear(24, 225),
            nn.ReLU(), 
            nn.Linear(225, 1),
            nn.Sigmoid() # binary classification problem 
        )
    def forward(self, x):
        x = self.flatten(x) 
        logits = self.linear_relu_stack(x) 
        return logits 

# now we will init and move the model itself 
model = NeuralNetwork().to(device) 
print(model) 

# architecture set up, now it seems like we can define the loss function and then make predictions 
loss_fn = nn.BCELoss()  
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3) 

# can now define the training loop 
# creating the data loaders for our training and testing data
# probably have to mess with my data somewhat before loading into pytorch

# doing some more pre-processing 
bool_cols = train.select_dtypes(include = ['bool']).columns 
# filtering 
train[bool_cols] = train[bool_cols].astype(int) 

# now doing this with the test data 
bool_cols = test.select_dtypes(include = ['bool']).columns 
test[bool_cols] = test[bool_cols].astype(int) 

# doing this for the objects 
#obj_cols = train.select_dtypes(include = ['object']).columns 
#train[obj_cols] = train[obj_cols].astype(int) 

# generating random numbers for a new customer ID column 
# rather than the alphanumeric string which is hard to work with 
customer_ids = train['customer_ID'].unique() 

# random numbers
np.random.seed(0) 
random_numbers = np.random.randint(200000, size = len(customer_ids)) 
print(random_numbers[:20])
print(len(random_numbers))

# making this list of random numbers into the first column of the dataframe 
id_to_random = dict(zip(customer_ids, random_numbers)) 
print(id_to_random) 
 
# now applying this to our training data 
train['customer_ID'] = train['customer_ID'].map(id_to_random) 

# verifying 
#print(train['customer_ID'].head(20)) 

# now doing this for the test data 
test_ids = test['customer_ID'].unique() 
np.random.seed(0) 
random_test = np.random.randint(200000 , size = len(test_ids)) 

id_to_random_test = dict(zip(test_ids, random_test)) 
print(id_to_random_test) 

test['customer_ID'] = test['customer_ID'].map(id_to_random_test) 

# verifying 
print(test['customer_ID'].head(20)) 

#assert train.dtypes.equals(test.dtypes) # should be the same 
print(train.dtypes) 
print(test.dtypes)
print(train.shape) 
print(test.shape)

# want to see which columns are in the train but not the test 
print(train.columns.difference(test.columns))

# need to convert the one object column to a float
# finding the object column 
# dropping the S_2 column 
train = train.drop(columns = ['S_2']) 
test = test.drop(columns = ['S_2']) 
obj_cols = train.select_dtypes(include = ['object']).columns 
print(obj_cols) 



# trying the conversion
# exporting!
train.to_feather('/home/adbucks/Documents/sta_629/Final_Project/nn_train_ready.feather') 
test.to_feather('/home/adbucks/Documents/sta_629/Final_Project/nn_test_ready.feather') 
train = torch.tensor(train.values, dtype = torch.float32) 
test = torch.tensor(test.values, dtype = torch.float32) 

print(train.shape) 
print(test.shape) 

train_dl = DataLoader(train, batch_size = batch_size, shuffle = True)  
test_dl = DataLoader(test, batch_size = batch_size) 

# now we can define our training loop 
# epochs same as the kaggle example 
epochs = 10 

# re-defining 
def train(model, train_loader, criterion, optimizer, device):
    model.train() 
    running_loss = 0.0 

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad() 
        outputs = model(inputs) 
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader) 
    print(f'Train Loss: {avg_loss:.4f}')

    return avg_loss 

# definig the test function 
def test(model, test_loader, criterion, device):
    model.eval() 
    running_loss = 0.0 
    correct = 0 
    total = 0 

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs) 
            loss = criterion(outputs, labels.unsqueeze(1).float())  
            running_loss += loss.item() 
            _, predicted = torch.max(outputs, 1) 
            total += labels.size(0) 
            correct += (predicted.cpu() == labels.cpu().unsqueeze(1)).sum().item()
    avg_loss = running_loss / len(test_loader) 
    accuracy = 100 * correct / total 
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%") 

    return avg_loss, accuracy

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}') 
    train_loss = train(model, train_dl, loss_fn, optimizer, device) 
    test_loss, test_accuracy = test(model, test_dl, loss_fn, device) 


# trying another way 
''' 

for epoch in range(epochs):
    for i in range(0, len(train), batch_size):
        X = train[i:i+batch_size] 
        y_pred = model(X)
        y = train[i:i+batch_size]
        loss = loss_fn(y_pred, y) 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()
        break
    print(f'Finished epoch {epoch}, latest loss {loss}') 
'''
# defining the training loop 
''' 
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) 
    model.train() 
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) 

        # compute pred error 
        pred = model(X) 
        loss = loss_fn(pred, y) 

        # backpropagation 
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad() 

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X) 
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset) 
    num_batches = len(dataloader) 
    model.eval() 
    test_loss, correct = 0, 0 
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) 
            pred = model(X) 
            test_loss += loss_fn(pred, y).item() 
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() 
    test_loss /= num_batches 
    correct /= size 
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") 


# now we can train with our given epochs 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer) 
    test(test_dl, model, loss_fn) 
print("Done!")
'''

