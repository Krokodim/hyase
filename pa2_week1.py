import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv')

# remove the rows with empty Age
data = data[np.isfinite(data['Age'])]

# convert Sex into integer
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

features = ['Sex','Age','Pclass','Fare']
label = 'Survived'

X = data[features].values
Y = data[label].values

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, Y)

print features
print clf.feature_importances_


