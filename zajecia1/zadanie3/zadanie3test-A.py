#!/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
flats = pd.read_csv('train/train.tsv', sep = '\t',
                   names = ['price', 'isNew','rooms', 'floor', 'location', 'sqrMetres'])
from sklearn import linear_model

reg = linear_model.LinearRegression()

X = flats['sqrMetres']
X = X.values.reshape(-1,1)
Y = flats['price']



model = reg.fit(X, Y)

a = model.coef_[0]
b = model.intercept_


print('y= {0}x + {1}'.format(a,b))

plt.scatter(X,Y, color='orange')
plt.plot([0,200],[b,a*200+b], 'r')
plt.title('Linear regression')
plt.xlabel('sqrMetres')
plt.ylabel('price')
plt.show() 

#Z = flats['sqrMetres','floor']

model2 = reg.fit(pd.DataFrame(flats, columns=['sqrMetres','floor']),flats['price'])

a = model2.coef_
b = model.intercept_

print('multi variable model: {0}, {1}, {2}'.format(a[0],a[1],b))

infile = pd.read_csv('test-A/in.tsv', sep='\t', names=['1','2','3','4','5'])
infile_correct = infile['5'].values
infile_correct = infile_correct.reshape(-1,1)
prediction = reg.predict(pd.DataFrame(infile, columns=['5','2']))

f = open('test-A/out.tsv', 'w')

for i in range(0, len(infile_correct)):
    f.write(str(prediction[i]) + '\n')

f.close()
