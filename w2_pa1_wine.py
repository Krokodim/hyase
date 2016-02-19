import pandas
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale

dt = pandas.read_csv('wine.data', header=None)

y = dt.ix[:,0].values
x = dt.ix[:,1:].values

kf = KFold(n = len(x),  n_folds = 5, shuffle = True, random_state=42)

max_cv1, max_cv1_k = 0, 0

for k in np.arange(1,51):
    knn = KNeighborsClassifier(n_neighbors=k)
    cv = cross_val_score(knn, x, y, cv=kf).mean()
    if cv > max_cv1: max_cv1, max_cv1_k = cv, k

print 'ORIGINAL: k = {1} cv={0}'.format(max_cv1, max_cv1_k)

x2 = scale(x)

max_cv2, max_cv2_k = 0, 0

for k in np.arange(1,51):
    knn = KNeighborsClassifier(n_neighbors=k)
    cv = cross_val_score(knn, x2, y, cv=kf).mean()
    if cv > max_cv2: max_cv2, max_cv2_k = cv, k

print 'SCALED: k = {1} cv={0}'.format(max_cv2, max_cv2_k)