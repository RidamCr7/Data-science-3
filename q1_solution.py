'''Question 1'''

# importing the important libraries
import pandas as pd
import sklearn as skit
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# reading the csv file
df = pd.read_csv('SteelPlateFaults-2class.csv')

# storing all data attribute except class
x = df.iloc[:, :df.shape[1] - 1]

# storing the class attribute data
y = df.iloc[:, df.shape[1] - 1]

"""Question 1"""

# Splitting the data in train test data
[X_train, X_test, X_label_train, X_label_test] = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)

# storing the train test data to external csv file
X_train.to_csv('SteelPlateFaults-train.csv')
X_test.to_csv('SteelPlateFaults-test.csv')

print("Before normalizing the result are : \n")

# making a loop that will compute confusion matrix, accuracy score for k = 1, 3, 5 respectively
for i in range(3):
    # using the inbuilt classifier
    classifier = KNeighborsClassifier(n_neighbors = 2*i + 1, metric = 'minkowski', p = 2)
    classifier.fit(X_train, X_label_train)
    
    # storing the predicted data by classifier
    y_pred = classifier.predict(X_test) 
    
    # printing the output
    print("\nFor K = ",2*i + 1 ," : ")
    print("\nconfusion matrix is -> \n",confusion_matrix(y_pred, X_label_test))
    print("\nAccuracy score is -> \n", accuracy_score(y_pred, X_label_test));
