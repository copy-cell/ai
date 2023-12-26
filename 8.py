from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix


iris = datasets.load_iris()
print("Iris data set loaded...\n")


x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1)
print("data set split into training and testing...")
print("size of training data and its label", x_train.shape, y_train.shape, "\n")
print("size of training data and its label", x_test.shape, y_test.shape, "\n")


for i in range(len(iris.target_names)):
    print("label", i, "-", str(iris.target_names[i]))


classifier = KNeighborsClassifier(n_neighbors=1)


classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


print("results of classification using k-nn with k=1")
for r in range(0, len(x_test)):
    print("sample: ", str(x_test[r]), "actual-label:", str(y_test[r]), "predicted label:", str(y_pred[r]), "\n")

print("classification accuracy:", classifier.score(x_test, y_test), "\n")

print("confusion matrix")
print(confusion_matrix(y_test, y_pred))
print("Accuracy matrix")
print(classification_report(y_test, y_pred))
