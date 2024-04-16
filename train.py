import torch
import torch.nn as nn
from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import ssl
from urllib.request import urlopen
from sklearn.model_selection import train_test_split

# Disable SSL certificate verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

torch.manual_seed(42)
model = Model()

url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'

# Read CSV file
with urlopen(url, context=ssl_context) as response:
    my_df = pd.read_csv(response)

print(my_df)

pd.set_option('future.no_silent_downcasting', True)
my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
my_df['variety'] = my_df['variety'].replace('Versicolor', 1.0)
my_df['variety'] = my_df['variety'].replace('Virginica', 2.0)

print("=====================================================================")
print(my_df)

X = my_df.drop('variety', axis=1)
y = my_df['variety']

# Convert to numpy arrays
X = X.values
y = y.values

# Train, test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(y_test)
print(type(y_test))
print(y_test.dtype)

# Convert float elements to integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Convert to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# Set the loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

print(model.parameters)

# TRAINING
epochs = 200
losses = []
for i in range(epochs+1):
    # Go forward and predict
    y_pred = model.forward(X_train)
    # Measure the loss/error
    loss = criterion(y_pred, y_train)
    # Keep track of losses
    losses.append(loss.detach().numpy())

    # print every 10 epochs
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.plot(range(epochs+1), losses)
plt.ylabel("Loss/error")
plt.xlabel("Epoch")
plt.show()

