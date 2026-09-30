import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier

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


X = df[df.columns[:-1]].values
Y = df[df.columns[-1]].values

X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.8)

scalar = StandardScaler()

print(X_train)
X_train_scaled = scalar.fit_transform(X_train)
print(X_train_scaled)
X_test_scaled = scalar.transform(X_test)


knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

yPred = knn_model.predict(X_test)
print(classification_report(y_true=y_test, y_pred=yPred))