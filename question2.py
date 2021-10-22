import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#Q2
df=pd.read_csv('SteelPlateFaults-2class.csv')

cols=list(df.columns)
cols.remove('Class')

df_normalise= df2
  
# apply normalization techniques
for column in  cols:
    df_normalise[column] = df_normalise[column]  / df_normalise[column].abs().max()
    
Y=df['Class'].to_numpy()

X = df_normalise. to_numpy()
print(X)

#K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)
a=list(knn.predict(X_test))
print(accuracy_score(Y_test, a))

print(confusion_matrix(a, Y_test))

#K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
a=list(knn.predict(X_test))
print(accuracy_score(Y_test, a))

print(confusion_matrix(a, Y_test))

#K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
a=list(knn.predict(X_test))
print(accuracy_score(Y_test, a))

print(confusion_matrix(a, Y_test))

#saving normalised train dataframe as csv file
df_train_norm = pd.DataFrame(X_train, columns =cols)
df_train_norm['Class']=Y_train
df_train_norm.to_csv('SteelPlateFaults-train-Normalised.csv')
print(df_train_norm)

#saving normalised test dataframe as csv file
df_test_norm = pd.DataFrame(X_test, columns =cols)
df_test_norm['Class']=Y_test
df_test_norm.to_csv('SteelPlateFaults-test-normalised.csv')
print(df_test_norm)

