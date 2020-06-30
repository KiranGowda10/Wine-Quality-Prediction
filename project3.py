#**********************************CECS 550 PATTERN RECONITION PROJECT-3****************************************
#                                       GROUP MEMBERS - Kiran M Gowda (018761559) 
#													   Karthik Ganduri (018779902)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

k=int(input("Enter the value of k: "))
cv1 = int(input("Enter the value of T ="))
R1= int(input("Enter the value of R1= "))
R2 = int(input("Enetr the value of R2 = "))


table_header = ['fixedacid', 'volacid', 'citricacid', 'residualsugar', 'chlorides', 'freesulfur', 'totalsulfur', 'density','pH', 'sulphates', 'alcohol', 'quality']   

data = pd.read_csv(r"C:\Users\KiranGM\Desktop\project-3\wine_quality.csv",header=None, names = table_header)
print(data)

data['new_quality'] = [0 if i <= 5 else 1 for i in data.quality]
print(data)

data['new_quality'].value_counts().plot(kind = 'bar', title = 'Number of classes in the dataset')

data1 = data.drop(columns = ['quality'])
print(data1)

data2 = data['new_quality']
print(data2) #class values

print(pd.DataFrame(data2.value_counts())) #no of high classes and low classes

X_train, X_test, y_train, y_test = train_test_split(data1, data2, test_size=0.2)
print(X_train) #trained splitted data


accuracy = {}
knn = KNeighborsClassifier(n_neighbors = k)
model = knn.fit(X_train,y_train)
pred = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test,pred)
print(accuracy) #accuracy 

kfold = KFold(n_splits=5, shuffle=False)
cv_score = cross_val_score(knn, data1, data2,cv = cv1, scoring='accuracy')
print(cv_score) #accuracy for each fold
print(cv_score.mean()) #mean accuracy

grid = KNeighborsClassifier()
grid1 = {"n_neighbors":np.arange(R1,R2)}
knn_gridcv = GridSearchCV(grid,grid1, cv = cv1)
knn_gridcv.fit(data1,data2)      #calculating accuracy for each value of k in the range 1-25

print(knn_gridcv.best_params_)  #the best value  of k

print(knn_gridcv.best_score_)  #the accuracy for gridsearchCV 






