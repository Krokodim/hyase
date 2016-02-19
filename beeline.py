import pandas
import numpy as np
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



raw = pandas.read_csv('train.csv')


X = raw.ix[:, :62]
y = raw.ix[:,  62].values


def xform (col): return scale(X[col].fillna(X[col].mean()))


df = pandas.DataFrame()

goodcols = [6,7]

for col in goodcols:
    colname = 'x{0}'.format(col)
    df[colname] = X[colname]



for i in range(62):
    if raw.dtypes[i] == 'float64':
        colname = 'x{0}'.format(i)
        df[colname] = xform(colname)

clf = KNeighborsClassifier()

clf.fit(df, y)

pred = clf.predict(df)

print accuracy_score(y, pred)