import numpy as np
#importing iris dataset which comes preloaded with sklearn
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
#creating a fold dataset for testing the classifier later
test_idx = [0,50,100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx ]

#training the classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target)
print (clf.predict(test_data))