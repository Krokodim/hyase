import numpy as np
from sklearn import      \
    datasets as ds,      \
    preprocessing as pp, \
    neighbors as nb,     \
    cross_validation

dt = ds.load_boston()

X = pp.scale(dt.data)
y = dt.target

names = dt.feature_names

knr = nb.KNeighborsRegressor(n_neighbors=5, weights='distance')

kf = cross_validation.KFold(n = len(X),  n_folds = 5, shuffle = True, random_state=42)

max_cv1, max_cv1_p = 0, 0

for p in np.linspace(1,10,200):
    cv = cross_validation.cross_val_score(knr, X, y, cv=kf).mean()
    if cv > max_cv1: max_cv1, max_cv1_p = cv, p

print 'p = {1} cv={0}'.format(max_cv1, max_cv1_p)

