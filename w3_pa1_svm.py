import pandas

from sklearn.svm import SVC


df = pandas.read_csv('svm-data.csv', header = None)

X = df.ix[:,1:]
y = df.ix[:,0]

clf = SVC(C=100000, kernel='linear')

clf.fit(X,y)

print clf.support_
