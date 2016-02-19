import pandas
import numpy as np
from scipy.stats.stats import pearsonr

import re


data = pandas.read_csv('titanic.csv', index_col='PassengerId')


print \
    'Total {0} passengers: {1} men, {2} women'.format(
            len(data),
            (data['Sex'] == 'male').sum(),
            (data['Sex'] == 'female').sum()
    )

print \
    '{0} passengers survived ({1}%)'.format(
        data['Survived'].sum(),
        100.0*data['Survived'].sum() / len(data)
    )

print \
    '{0} 1st class passengers  ({1}%)'.format(
        (data['Pclass']==1).sum(),
        100.0 * (data['Pclass']==1).sum() / len(data)
    )


print \
    'Mean age: {0} Age median: {1}'.format(
        np.nanmean(data['Age']),
        np.nanmedian(data['Age'])
    )

print 'Pearson for SibSp ~ Parch = {0}'.format(
        pearsonr(data['SibSp'], data['Parch'])[0]
)

names = data[data['Sex']=='female']['Name']
nd = dict()

for name in names:
    first_name = name.split('.')[1].strip()
    rr = re.search(r'\(([A-Z a-z]+)\)', first_name)
    if rr: first_name = rr.group(1)
    first_name = first_name.split(' ')[0]
    try:
        nd[first_name] += 1
    except:
        nd[first_name] = 1

print sorted(nd.items(), key=lambda x:x[1], reverse = True)[0:5]


