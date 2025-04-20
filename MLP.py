# %%
import pandas as pd

df = pd.read_csv("preprocessed_dataset.csv",nrows=1000000)
print(df.head())

# %%
import torch
from sklearn.model_selection import train_test_split

# Separate features and labels
X = df.drop(columns=['attack'])  # Features
y = df['attack']  # Target (0 = Benign, 1 = Attack)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

print("Data prepared for Deep Learning!")

# %%
import torch.nn as nn
import torch.optim as optim

# Define MLP model
class DDoSClassifier(nn.Module):
    def __init__(self, input_size):
        super(DDoSClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 2)  # Binary Classification (0 = Benign, 1 = Attack)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Initialize model
input_size = X_train.shape[1]  # Number of features
model = DDoSClassifier(input_size)

print(model)


# %%
# Loss function & optimizer
criterion = nn.CrossEntropyLoss()  # Since we have two classes (0,1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Loss function & optimizer defined!")


# %%
# Training loop
epochs = 30  # Start with 30 epochs, increase if needed
batch_size = 1024
train_loader = torch.utils.data.DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

print("Model training complete!")


# %%
# Evaluate on test data
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    y_pred_labels = torch.argmax(y_pred_test, axis=1)

# Calculate accuracy
accuracy = (y_pred_labels == y_test_tensor).float().mean().item()
print(f"Test Accuracy: {accuracy:.4f}")



