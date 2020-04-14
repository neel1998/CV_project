import numpy as np
import pickle 
import os
from utils import NUM_OF_WORDS, TRAIN_PATH
from torch import nn, optim
import torch

print("import finished")
def freq(curr_labels):
	hist = np.bincount(curr_labels[:, 0], minlength=NUM_OF_WORDS)
	return hist


class MLP(nn.Module):
    
    def __init__(self):
        super(MLP, self).__init__()
        
        n_out = 2
        self.linear = nn.Sequential(
            
            # nn.Linear(4000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 512),
            # nn.ReLU(),
            nn.Linear(4000, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, n_out)
        )

    def forward(self, x):
        x = self.linear(x)
        return x

# Generating the data
with open('dict.pkl', 'rb') as f:
	DICT = pickle.load(f)

with open('labels.pkl', 'rb') as f:
	labels = pickle.load(f)

X = []
Y = []

csum = 0
for i, folder in enumerate(sorted(os.listdir(TRAIN_PATH))):
    
    # (A): the native direction of the video
    curr_labels = labels[csum:csum+DICT[folder]['A']]
    csum += DICT[folder]['A']
    X.append(freq(curr_labels))
    if folder[0] == 'F':
    	Y.append(1)
    else:
    	Y.append(0)
    
    # (B): this video mirrored in the left-right direction
    curr_labels = labels[csum:csum+DICT[folder]['B']]
    csum += DICT[folder]['B']
    X.append(freq(curr_labels))
    if folder[0] == 'F':
    	Y.append(1)
    else:
    	Y.append(0)

    # (C): the original video time-flipped;
    curr_labels = labels[csum:csum+DICT[folder]['C']]
    csum += DICT[folder]['C']
    X.append(freq(curr_labels))
    if folder[0] == 'F':
    	Y.append(0)
    else:
    	Y.append(1)

    # (D): the time-flipped left-right-mirrored version.
    curr_labels = labels[csum:csum+DICT[folder]['D']]
    csum += DICT[folder]['D']
    X.append(freq(curr_labels))
    if folder[0] == 'F':
    	Y.append(0)
    else:
    	Y.append(1)

X = np.array(X)
Y = np.array(Y)

# Normalizing features
X = X/(np.linalg.norm(X, axis=1)[:, None]+1e-10)
print('X.shape:', X.shape, 'Y.shape:', Y.shape)

mlp = MLP().double()

loss = nn.CrossEntropyLoss()
# loss = nn.BCELoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.1)


idx = np.arange(X.shape[0])
idx = np.random.shuffle(idx)

X = X[idx]
Y = Y[idx]

X = torch.from_numpy(X)[0].double()
Y = torch.from_numpy(Y)[0]

print('X.shape:', X.shape, 'Y.shape:', Y.shape)

train_size = 408

X_train = X[:train_size,:]
Y_train = Y[:train_size]

X_val = X[train_size:,:]
Y_val = Y[train_size:]

print(X_train.size(0), X_train.size(1))

for epoch in range(50):
    
    running_loss = 0.0
    total = 0
    corr = 0
    batch_size = 8
    for i in range(0,train_size,batch_size):

        inputs = X_train[i:i+batch_size,:]
        labels = Y_train[i:i+batch_size]

        # print(inputs.size(0), inputs.size(1))
        optimizer.zero_grad()
        outputs = mlp(inputs.double())
        # print("Outputs: ", outputs)       
        pred_err = loss(outputs, labels)
        pred_err.backward()
        optimizer.step()

        running_loss += pred_err.item()
        _,pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        corr += (pred == labels).sum().item()
    
    if (epoch+1)%10 == 0:
        torch.save(mlp.state_dict(), './models/mlp_SGD_01_' + str(epoch + 1) + '.pth')

    acc = corr/total
    s = "After epoch " + str(epoch + 1) + " Training Accuracy: " + str(acc) + " Loss: " + str(running_loss)
    print(s)
    #Validation
    outputs = mlp(X_val)
    pred_err = loss(outputs, Y_val)
    _,pred = torch.max(outputs.data, 1)
    val_acc = (pred == Y_val).sum().item()/Y_val.shape[0]
    s = "After epoch " + str(epoch + 1) + " Validation Accuracy: " + str(val_acc)
    print(s)
