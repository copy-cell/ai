import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('6.csv')

print("The first 5 Values of data are:\n", data.head())

X = data.iloc[:, :-1]
print("\nThe First 5 values of the train data are\n", X.head())

y = data.iloc[:, -1]
print("\nThe First 5 values of train output are\n", y.head())

categorical_columns = ['Outlook', 'Temperature', 'Humidity', 'Windy']

X = pd.get_dummies(X, columns=categorical_columns)

print("\nNow the Train data is\n", X.head())

le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)

print("\nNow the Train output is\n", y)

random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_seed)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

accuracy = accuracy_score(classifier.predict(X_test), y_test)
print("Accuracy is:", accuracy)
