#! /bin/python3

import pandas as pd

flats = pd.read_csv('train/train.tsv', sep = '\t', 
        names = ['price', 'isNew', 'rooms', 'floor', 'location', 'sqrMeters'])

infile = pd.read_csv('dev-0/in.tsv', sep='\t')

f = open('dev-0/out.tsv', 'w')
for i in range (0,len(infile)):
    f.write(str(flats.price.mean()) + '\n')

f.close()


