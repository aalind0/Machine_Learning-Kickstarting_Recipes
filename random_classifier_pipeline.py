import random

#writing a random classifier
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions


#import the dataset
from sklearn import datasets
iris = datasets.load_iris()

#naming the feature and the test_target
X = iris.data
y = iris.target

#creating multifold partitions of the training data only into train and test partitions.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

#implementing the decesion tree classifier
#from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

#calling the predict method
predictions = my_classifier.predict(X_test)
print(predictions)

#calculating the accuracy of our classifier by matching the train data with the test data
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))