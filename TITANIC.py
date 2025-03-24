import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import kagglehub

path = kagglehub.dataset_download("ibrahimelsayed182/titanic-dataset")
file_path = f"{path}/Titanic.csv"  

scaler = StandardScaler()

df = pd.read_csv(file_path, usecols=['sex', 'age', 'sibsp', 'parch', 'embarked', 'class', 'who', 'alone'])
df2 = pd.read_csv(file_path, usecols=['survived'])

df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["embarked"] = df["embarked"].map({"S": 1, "C": 0, "Q": 2})
df["class"] = df["class"].map({"First": 0, "Second": 1, "Third": 2})
df["who"] = df["who"].map({"man": 0, "woman": 1, "child": 2})
df["alone"] = df["alone"].astype(int)
df2["survived"] = df2['survived'].astype(int)
df["age"] = df["age"].fillna(df["age"].mean())
df = df.fillna(0)
df2 = df2.fillna(0)
print(df.dtypes)

df[['sex', 'age', 'sibsp', 'parch', 'embarked', 'class', 'who', 'alone']] = scaler.fit_transform(df[['sex', 'age', 'sibsp', 'parch', 'embarked', 'class', 'who', 'alone']])

X_train, X_test, y_train, y_test = train_test_split(
    df.values, df2.values.flatten(), test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class Titanic(nn.Module) :
    def __init__(self) :
        super(Titanic, self).__init__()
        self.hidden1 = nn.Linear(8, 128)
        self.act = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.05)
        self.hidden2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.05)
        self.hidden3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)
    def forward(self, x) :
        x = self.act(self.hidden1(x))
        x = self.dropout1(x)
        x = self.act(self.hidden2(x))
        x = self.dropout2(x)
        x = self.act(self.hidden3(x))
        return x
    
model = Titanic()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)

print("Min:", torch.min(X_train))
print("Max:", torch.max(X_train))
print("Any NaN:", torch.isnan(X_train).any())

epochs = 30000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"Loss : {loss.item():.4f}, lr : {lr_now:.6f}")
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

with torch.no_grad():
    test = [[1, 30, 0, 0, 1, 1, 1, 1]]
    test_scaled = scaler.transform(test)
    test_input = torch.tensor(test_scaled, dtype=torch.float32)
    output = model(test_input)
    pred = torch.argmax(output, dim=1).item()
    print(f"Prédiction : {'survie' if pred == 1 else 'mort'}")


with torch.no_grad():
    model.eval()
    y_pred_test = model(X_test)
    predicted = torch.argmax(y_pred_test, dim=1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")