import numpy as np
import pandas as pd

input_file = "sonar.all-data"


# comma delimited is the default
df = pd.read_csv(input_file, header = None)
#target_class = Dataset[df.columns[len(Dataset.axes[1])-1:]]
# remove the non-numeric columns
feature = df.select_dtypes(include=['float64']);
lable = df.select_dtypes(exclude=['float64']);


#df.drop(df.columns[[60]], axis=1)  # df.columns is zero-based pd.Index
numpy_array1 = feature.as_matrix()
numpy_array2 = lable.as_matrix()

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV


X_train, X_test, Y_train, Y_test = train_test_split(numpy_array1, numpy_array2, test_size=0.5)
clf = DecisionTreeClassifier();
y = Y_train.ravel()
Y_train = np.array(y).astype(str)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
num_folds = 10
seed = 7
scoring = 'accuracy'

#clf.fit(X_train, y_train)
#prediction = clf.predict(X_test)
#print(prediction.size)
#y = y_test.ravel()
#y_test = np.array(y).astype(str)
#print(y_test.size);


#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test,prediction))
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
