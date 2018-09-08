#!/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cars = pd.read_csv('train/in.tsv', sep = '\t',
                   names = ['price', 'mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])
    
cars_cleared = cars[cars.price>1000]

from sklearn import linear_model

reg = linear_model.LinearRegression()

model = reg.fit(pd.DataFrame(cars_cleared, columns=['year','mileage','engineCapacity']),cars_cleared['price'])

a = model.coef_
b = model.intercept_

print('multi variable model: {0}, {1}, {2}, {3}'.format(a[0],a[1],a[2],b))

infile = pd.read_csv('test-A/in.tsv', sep='\t', names=['mileage','year','brand','engineType','engineCapacity'])

prediction = reg.predict(pd.DataFrame(infile, columns=['year','mileage', 'engineCapacity']))

f = open('test-A/out.tsv', 'w')

for i in range(0, len(prediction)):
    f.write(str(prediction[i]) + '\n')

f.close()
