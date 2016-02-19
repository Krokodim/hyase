import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df_train = pandas.read_csv('perceptron-train.csv', header = None)
y_train  = df_train.ix[:,0].values
x_train  = df_train.ix[:,1:].values

df_test = pandas.read_csv('perceptron-test.csv', header = None)

y_test  = df_test.ix[:,0].values
x_test  = df_test.ix[:,1:].values


clf = Perceptron(random_state=241)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

a1 = accuracy_score(y_test, y_pred)

print a1

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

clf.fit(x_train_scaled, y_train)

y_pred = clf.predict(x_test_scaled)

a2 = accuracy_score(y_test, y_pred)

print a2


print a2-a1
