import warnings 
warnings.simplefilter('ignore') 
import pandas as pd 
import numpy as np
import torch 
from torch import nn 
import os 
from torch.utils.data import DataLoader, TensorDataset 

# loading in our feathered data 
train = pd.read_feather('/home/adbucks/Documents/sta_629/Final_Project/nn_train_ready.feather') 
test = pd.read_feather('/home/adbucks/Documents/sta_629/Final_Project/nn_test_ready.feather') 

# let's see what the data looks like 
# dropping a few columns from train and adding the labels to test 
#train = train.drop(['oneHot_D_64_1', 'oneHot_D_66_0.0', 'oneHot_D_68_0.0'])

# can actually just split the train data and make that the new test...
from sklearn.model_selection import train_test_split 

train, test = train_test_split(train, test_size = 0.2) 

# now we can see what the data looks like 
print(train.head()) 
print(train.shape) 
print(test.head()) 
print(test.shape)
print(train['target'].value_counts())
print(type(train['target'])) 

# needs to be changed to float32 
train['target'] = train['target'].astype('float32') 
test['target'] = test['target'].astype('float32') 

# now adding the labels to test 


print(type(train)) 

# more feature engineering 
X = torch.tensor(train.drop('target', axis = 1).values, dtype = torch.float32) 
y = torch.tensor(train['target'].values, dtype = torch.float32) 

dataset = TensorDataset(X, y) 

batch_size = 224 
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# trying to do this with the test data 
X_test = torch.tensor(test.drop('target', axis = 1).values, dtype = torch.float32) 
y_test = torch.tensor(test['target'].values, dtype = torch.float32) 

ttest = TensorDataset(X_test, y_test) 
ttest_loader = DataLoader(test, batch_size=batch_size, shuffle=False) 

# let's see what the data looks like 
print(X.shape) 
print(y.shape) 
print(X_test.shape) 
print(y_test.shape)
print(type(y))  

print(train_loader) 
print(ttest_loader) 


# now we'll set up the model architecture 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() 
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(224, 24), 
            nn.ReLU(), 
            nn.Linear(24, 224), 
            nn.ReLU(), 
            nn.Linear(224, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x) 
        logits = self.linear_relu_stack(x) 
        return logits

device = (
    "cuda"
    if torch.cuda.is_available() 
    else "mps" 
    if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"Using {device} device") 

model = NeuralNetwork().to(device) 
print(model) 

loss_fn = nn.BCELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# now can define training and testing 
def train(model, loss_fn, optimizer, train_loader, device):
    model.train() 
    running_loss = 0.0 

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) 

        optimizer.zero_grad() 

        outputs = model(inputs) 
        loss = loss_fn(outputs, labels.unsqueeze(1).float())  

        # backward pass and optimization 
        loss.backward() 
        optimizer.step() 

        running_loss += loss.item() 
    avg_loss = running_loss / len(train_loader) 

    print(f'Loss: {avg_loss:.4f}')

    return avg_loss 
    
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

            # get the accuracy 
            predicted = torch.round(outputs) 
            total += labels.size(0) 
            correct += (predicted.cpu() == labels.cpu().unsqueeze(1).sum().item()) 
    avg_loss = running_loss / len(test_loader) 
    accuracy = 100 * correct / total 
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy 


# now we can train the model 
epochs = 10 
batchsize = 224

for e in range(epochs):
    print(f'Epoch {e + 1}/{epochs}')
    train_loss = train(model, loss_fn, optimizer, train_loader, device) 
    test_loss, test_accuracy = test(model, ttest_loader, loss_fn, device) 

