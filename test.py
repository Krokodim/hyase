import pandas
import numpy as np

df = pandas.DataFrame()

df['a']  = [1,2,3,np.nan]
df['b'] = [1,4,5,7]

df['a'] = df['a'].fillna(df['a'].mean())

print (df)

print 'x{0}'.format(14)