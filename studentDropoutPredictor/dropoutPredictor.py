import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import torch as t
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

os.chdir("studentDropoutPredictor")

def onehotEncode(dataframe: pd.DataFrame, coloumn: str) -> pd.DataFrame:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(dataframe[[coloumn]]) 

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out([coloumn]),
        index = dataframe.index
    )

    dataframe = pd.concat([
        dataframe.drop(columns=[coloumn]),
        encoded_df
     ], axis=1)

    return dataframe



def maxNormalisation(dataframe: pd.DataFrame, column:str) -> pd.DataFrame:
    colMax = dataframe[column].max()
    dataframe[column] = dataframe[column].div(colMax)
    return dataframe

df = pd.read_csv("student dropout.csv")

df["School"] = (df["School"] == "GP").astype(int) # 1 means the school is GP, 0 means the school is MS
df["Gender"] = (df["Gender"] == "M").astype(int) # 1 means the person is male, 0 means the person is female
df["Address"] = (df["Address"] == "U").astype(int) # 1 means the person is from an urban location, 0 means the person is from an rural location
df["Family_Size"] = (df["Family_Size"] == "GT3").astype(int) # 1 means the person's family size is 3 or more, 0 means the person's family is less than 3
df["Parental_Status"] = (df["Parental_Status"] == "A").astype(int) # 1 means the person's parents are living together, 0 means the person's parents live apart


df = onehotEncode(df, "Mother_Job")
df = onehotEncode(df, "Father_Job")
df = onehotEncode(df, "Reason_for_Choosing_School")
df = onehotEncode(df, "Guardian")

# Need to clean up the following columns: travel_time, study_time and number of failures
# The following columns are self-explanatory, 1 means yes and 0 means no
df["School_Support"] = (df["School_Support"] == "yes").astype(int) 
df["Family_Support"] = (df["Family_Support"] == "yes").astype(int) 
df["Extra_Paid_Class"] = (df["Extra_Paid_Class"] == "yes").astype(int) 
df["Extra_Curricular_Activities"] = (df["Extra_Curricular_Activities"] == "yes").astype(int) 
df["Attended_Nursery"] = (df["Attended_Nursery"] == "yes").astype(int) 
df["Wants_Higher_Education"] = (df["Wants_Higher_Education"] == "yes").astype(int) 
df["Internet_Access"] = (df["Internet_Access"] == "yes").astype(int) 
df["In_Relationship"] = (df["In_Relationship"] == "yes").astype(int) 
df["Dropped_Out"] = (df["Dropped_Out"] == True).astype(int)

df = df.assign(DroppedOut = df.pop("Dropped_Out"))

def showPlots():
    for label in df.columns[:-1]:
        plt.hist(df[df["DroppedOut"]==1][label], color='red', label='Dropped Out', alpha=0.7, density=True)
        plt.hist(df[df["DroppedOut"]==0][label], color='blue', label='In School', alpha=0.7, density=True)
        plt.title(label)
        plt.ylabel("Probability")
        plt.xlabel(label)
        plt.legend()
        plt.show()


showPlots()

X = df[df.columns[:-1]].values
Y = df[df.columns[-1]].values

X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.8)

scalar = StandardScaler()

print(X_train)
X_train_scaled = scalar.fit_transform(X_train)
print(X_train_scaled)
X_test_scaled = scalar.transform(X_test)

X_train_scaled_tensor = t.from_numpy(X_train_scaled).float()
X_test_scaled_tensor = t.from_numpy(X_test_scaled).float()
Y_train_tensor = t.from_numpy(y_train).float().unsqueeze(1)
Y_test_tensor = t.from_numpy(y_test).float().unsqueeze(1)

train_dataset = TensorDataset(X_train_scaled_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class BCNet(nn.Module):
    def __init__(self):
        super(BCNet, self).__init__()

        self.fc1 = nn.Linear(46, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc3(x))

        return x 


model = BCNet()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20


for epoch in range(epochs):
    model.train()
    runningLoss = 0.0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()

        preds = model(x_batch)
        loss = criterion(preds, y_batch)

        

        loss.backward()
        optimizer.step()

        runningLoss += loss.item()

    #print(f'Epoch {epoch+1}: Loss was {runningLoss / len(train_loader)}')



with t.no_grad():
    model.eval()

    preds = model(X_test_scaled_tensor)
    loss = criterion(preds, Y_test_tensor).item()

    accuracy = ((preds >= 0.5) == Y_test_tensor).float().mean().item()

print(accuracy)
