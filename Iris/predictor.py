import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

os.chdir("Iris")

df = pd.read_csv("IRIS.csv")



def showPlots():
    plt.scatter(df[df["species"]=="Iris-setosa"]["sepal_length"], df[df["species"]=="Iris-setosa"]["sepal_width"], c="r", label="Iris-setosa")
    plt.scatter(df[df["species"]=="Iris-versicolor"]["sepal_length"], df[df["species"]=="Iris-versicolor"]["sepal_width"], c="g", label="Iris-versicolor")
    plt.scatter(df[df["species"]=="Iris-virginica"]["sepal_length"], df[df["species"]=="Iris-virginica"]["sepal_width"], c="b", label="Iris-virginica")
    plt.title("Sepal Length / Sepal Width")
    plt.ylabel("sepal_width")
    plt.xlabel("sepal_length")
    plt.legend()
    plt.show()

    plt.scatter(df[df["species"]=="Iris-setosa"]["petal_length"], df[df["species"]=="Iris-setosa"]["petal_width"], c="r", label="Iris-setosa")
    plt.scatter(df[df["species"]=="Iris-versicolor"]["petal_length"], df[df["species"]=="Iris-versicolor"]["petal_width"], c="g", label="Iris-versicolor")
    plt.scatter(df[df["species"]=="Iris-virginica"]["petal_length"], df[df["species"]=="Iris-virginica"]["petal_width"], c="b", label="Iris-virginica")
    plt.title("Petal Length / Petal Width")
    plt.ylabel("Petal_width")
    plt.xlabel("Petal_length")
    plt.legend()
    plt.show()


X = df[df.columns[:-1]].values
Y = df[df.columns[-1]].values

X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.8)

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

yPred = knn_model.predict(X_test)
print(classification_report(y_true=y_test, y_pred=yPred))

