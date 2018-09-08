import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

train_data = pd.read_csv('train/train.tsv', sep='\t',
        names = ['Occupancy', 'date', 'Temperature', 'Humidity', 'Light',
            'CO2', 'HumidityRatio'])

lr = LogisticRegression()
lr.fit(train_data.CO2.values.reshape(-1,1), train_data.Occupancy)

f = open('notes', 'w')

#Dane treningowe
f.write('----DANE TRENINGOWE----- \n')
f.write('Rozkład próby treningowej: ')
f.write(str(sum(train_data.Occupancy == 1) / len(train_data.Occupancy)))
f.write('\n')
#print('Dokładność algorytmu zero rule: ')
#print( 1- sum(train_data.Occupancy == 1) / len(train_data.Occupancy))
#TP
TP = sum((lr.predict(train_data.CO2.values.reshape(-1,1)) == train_data.Occupancy) & (lr.predict(train_data.CO2.values.reshape(-1,1)) == 1))
f.write('True Positives: ')
f.write(str(TP))
f.write('\n')
#TN
TN = sum((lr.predict(train_data.CO2.values.reshape(-1,1)) == train_data.Occupancy) & (lr.predict(train_data.CO2.values.reshape(-1,1))     == 0))
f.write('True Negatives: ')
f.write(str(TN))
f.write('\n')
#FP
FP = sum((lr.predict(train_data.CO2.values.reshape(-1,1)) != train_data.Occupancy) & (lr.predict(train_data.CO2.values.reshape(-1,1))     == 1))
f.write('False Positives: ')
f.write(str(FP))
f.write('\n')
#FN
FN = sum((lr.predict(train_data.CO2.values.reshape(-1,1)) != train_data.Occupancy) & (lr.predict(train_data.CO2.values.reshape(-1,1))     == 0))
f.write('False Negatives: ')
f.write(str(FN))
f.write('\n')

f.write('----------Macierz błędu---------- \n')
f.write(str(confusion_matrix(lr.predict(train_data.CO2.values.reshape(-1,1)), train_data.Occupancy)))
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

dev_data = pd.read_csv('dev-0/in.tsv', sep='\t',
        names = ['date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])

dev_data_expected = pd.read_csv('dev-0/expected.tsv', sep='\t', names = ['y'])


f.write('----DANE DEWELOPERSKIE----- \n')
f.write('zero rule na zbiorze deweloperskim: ')
f.write(str(1 - sum(lr.predict(dev_data.CO2.values.reshape(-1,1))) / len (dev_data.CO2)))
f.write('\n')
#TP
TP = sum((lr.predict(dev_data.CO2.values.reshape(-1,1)) == dev_data_expected.y) & (lr.predict(dev_data.CO2.values.reshape(-1,1)) == 1))
f.write('True Positives: ')
f.write(str(TP))
f.write('\n')
#TN
TN = sum((lr.predict(dev_data.CO2.values.reshape(-1,1)) == dev_data_expected.y) & (lr.predict(dev_data.CO2.values.reshape(-1,1))     == 0))
f.write('True Negatives: ')
f.write(str(TN))
f.write('\n')
#FP
FP = sum((lr.predict(dev_data.CO2.values.reshape(-1,1)) != dev_data_expected.y) & (lr.predict(dev_data.CO2.values.reshape(-1,1))     == 1))
f.write('False Positives: ')
f.write(str(FP))
f.write('\n')
#FN
FN = sum((lr.predict(dev_data.CO2.values.reshape(-1,1)) != dev_data_expected.y) & (lr.predict(dev_data.CO2.values.reshape(-1,1))     == 0))
f.write('False Negatives: ')
f.write(str(FN))
f.write('\n')

f.write('----------Macierz błędu---------- \n')
f.write(str(confusion_matrix(lr.predict(dev_data.CO2.values.reshape(-1,1)), dev_data_expected.y)))
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

sns.regplot(x=dev_data.CO2, y=dev_data_expected.y, logistic=True, y_jitter=.1)
#plt.show()
plt.savefig('wykres_regresji_log')

prediction = lr.predict(dev_data.CO2.values.reshape(-1,1))

outfile = open('dev-0/out.tsv', 'w')
for i in range(0, len(prediction)):
    outfile.write(str(prediction[i]) + '\n')

outfile.close()

outfile_test = open('test-A/out.tsv', 'w')

testfile = pd.read_csv('test-A/in.tsv', sep='\t', names = ['date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])

prediction2 = lr.predict(testfile.CO2.values.reshape(-1,1))

for i in range(0, len(prediction2)):
    outfile_test.write(str(prediction2[i]) + '\n')

outfile_test.close()
