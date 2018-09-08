import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

train_data = pd.read_csv('train/in.tsv', sep='\t', header=None)
X = pd.DataFrame(train_data.loc[:, 1:])
lr_full = LogisticRegression()
lr_full.fit(X, train_data[0])

f = open('notes', 'w')

#Dane treningowe
f.write('----DANE TRENINGOWE----- \n')
f.write('Rozkład próby treningowej: ')
f.write(str(sum(train_data[0] == 'g') / len(train_data[0])))
f.write('\n')
#print('Dokładność algorytmu zero rule: ')
#print( 1- sum(train_data.Occupancy == 1) / len(train_data.Occupancy))
#TP
TP = sum((lr_full.predict(X) == train_data[0]) & (lr_full.predict(X) == 'g'))
f.write('True Positives: ')
f.write(str(TP))
f.write('\n')
#TN
TN = sum((lr_full.predict(X) == train_data[0]) & (lr_full.predict(X)     == 'b'))
f.write('True Negatives: ')
f.write(str(TN))
f.write('\n')
#FP
FP = sum((lr_full.predict(X) != train_data[0]) & (lr_full.predict(X)     == 'g'))
f.write('False Positives: ')
f.write(str(FP))
f.write('\n')
#FN
FN = sum((lr_full.predict(X) != train_data[0]) & (lr_full.predict(X) == 'b'))
f.write('False Negatives: ')
f.write(str(FN))
f.write('\n')

f.write('----------Macierz błędu---------- \n')
f.write(str(confusion_matrix(lr_full.predict(X), train_data[0])))
f.write('\n')
f.write('Accuracy, specificity and sensitivity \n')
ACC = (TP + TN) / (TP + FP + FN + TN)
SPC = TN / (TN + FP)
TPR =  TP / (TP + FN)
f.write('Accuracy: ')
f.write(str(ACC))
f.write('\n')
f.write('Specificity: ')
f.write(str(SPC))
f.write('\n')
f.write('Sensitivity: ')
f.write(str(TPR))
f.write('\n')

#sns.regplot(x=train_data.CO2, y=train_data.Occupancy, logistic=True, y_jitter=.1)
#plt.show()




#Dane deweloperskie

dev_data = pd.read_csv('dev-0/in.tsv', sep='\t', header=None)

dev_data_expected = pd.read_csv('dev-0/expected.tsv', sep='\t', names = ['y'])


f.write('----DANE DEWELOPERSKIE----- \n')
f.write('zero rule na zbiorze deweloperskim: ')
f.write(str(1 - sum(dev_data_expected.y == 'g') / len(dev_data)))
f.write('\n')
#TP
TP = sum((lr_full.predict(dev_data) == dev_data_expected.y) & (lr_full.predict(dev_data) == 'g'))
f.write('True Positives: ')
f.write(str(TP))
f.write('\n')
#TN
TN = sum((lr_full.predict(dev_data) == dev_data_expected.y) & (lr_full.predict(dev_data) == 'b'))
f.write('True Negatives: ')
f.write(str(TN))
f.write('\n')
#FP
FP = sum((lr_full.predict(dev_data) != dev_data_expected.y) & (lr_full.predict(dev_data) == 'g'))
f.write('False Positives: ')
f.write(str(FP))
f.write('\n')
#FN
FN = sum((lr_full.predict(dev_data) != dev_data_expected.y) & (lr_full.predict(dev_data) == 'b'))
f.write('False Negatives: ')
f.write(str(FN))
f.write('\n')

f.write('----------Macierz błędu---------- \n')
f.write(str(confusion_matrix(lr_full.predict(dev_data), dev_data_expected.y)))
f.write('\n')
f.write('Accuracy, specificity and sensitivity \n')
ACC = (TP + TN) / (TP + FP + FN + TN)
SPC = TN / (TN + FP)    
TPR =  TP / (TP + FN)
f.write('Accuracy: ')
f.write(str(ACC))
f.write('\n')
f.write('Specificity: ')
f.write(str(SPC))
f.write('\n')
f.write('Sensitivity: ')
f.write(str(TPR))
f.write('\n')


f.close()
#data_to_plot = dev_data_expected.replace(to_replace='g', value=1)
#data_to_plot = data_to_plot.replace(to_replace='b', value=0)
#sns.regplot(x=dev_data, y=data_to_plot.y, logistic=True, y_jitter=.1)
#plt.show()
plt.savefig('wykres_regresji_log')

prediction = lr_full.predict(dev_data)

outfile = open('dev-0/out.tsv', 'w')
for i in range(0, len(prediction)):
    outfile.write(str(prediction[i]) + '\n')

outfile.close()

outfile_test = open('test-A/out.tsv', 'w')

testfile = pd.read_csv('test-A/in.tsv', sep='\t', header=None)

prediction2 = lr_full.predict(testfile)

for i in range(0, len(prediction2)):
    outfile_test.write(str(prediction2[i]) + '\n')

outfile_test.close()
