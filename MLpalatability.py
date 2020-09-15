import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


df = pd.read_csv('Datasets/datasetC.csv')  # dataset upload
max_palatability = df['lable'].max()  # maximum value of palatability inside the dataset
dataset = df.copy()

'''
    reversing the palatability value in order to have higher values for better food baskets
'''
dataset['lable'] = max_palatability - dataset['lable']


features_name = ['Beans', 'Bulgur', 'Cheese', 'Fish', 'Meat', 'Corn-soya blend (CSB)', 'Dates',
                 'Dried skim milk (enriched) (DSM)', 'Milk', 'Salt', 'Lentils', 'Maize', 'Maize meal', 'Chickpeas',
                 'Rice',
                 'Sorghum/millet', 'Soya-fortified bulgur wheat', 'Soya-fortified maize meal',
                 'Soya-fortified sorghum grits', 'Soya-fortified wheat flour', 'Sugar', 'Oil', 'Wheat', 'Wheat flour',
                 'Wheat-soya blend (WSB)']

target_name = ['lable']

# Scaling the labels
sc_t = MinMaxScaler(feature_range=(0, 10))
dataset[target_name] = sc_t.fit_transform((dataset[target_name]))
X = dataset[features_name]
y = dataset[target_name]

# Splitting the dataset in training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

### LINEAR REGRESSION ###
# clf = LinearRegression(fit_intercept=True).fit(X_train, y_train)
# filename = 'models/linearRegression_W.sav'
# prediction_linear = clf.predict(X_test)
file_lr = 'models/linearRegression_W.sav'
linear_reg = pickle.load(open(file_lr, 'rb'))
prediction_linear = linear_reg.predict(X_test)

# Saving the trained model
# pickle.dump(clf, open(filename, 'wb'))

# Final check on the accuracy

counter = 0
counter_8 = 0
counter_9 = 0
counter_7 = 0
counter_6 = 0
counter_5 = 0
counter_4 = 0
counter_3 = 0
counter_2 = 0
counter_1 = 0
counter_0 = 0

for i in range(0, len(prediction_linear)):
    if abs((y_test[i:i + 1]['lable'] - prediction_linear[i][0]).values[0]) >= 0.5:
        counter += 1
        if((prediction_linear[i][0] >= 8) & (prediction_linear[i][0] < 9)):
            counter_8 += 1
        elif((prediction_linear[i][0] >= 9) & (prediction_linear[i][0] < 10)):
            counter_9 += 1
        elif((prediction_linear[i][0] >= 7) & (prediction_linear[i][0] < 8)):
            counter_7 += 1
        elif((prediction_linear[i][0] >= 6) & (prediction_linear[i][0] < 7)):
            counter_6 += 1
        elif((prediction_linear[i][0] >= 5) & (prediction_linear[i][0] < 6)):
            counter_5 += 1
        elif((prediction_linear[i][0] >= 4) & (prediction_linear[i][0] < 5)):
            counter_4 += 1
        elif((prediction_linear[i][0] >= 3) & (prediction_linear[i][0] < 4)):
            counter_3 += 1
        elif((prediction_linear[i][0] >= 2) & (prediction_linear[i][0] < 3)):
            counter_2 += 1
        elif ((prediction_linear[i][0] >= 1) & (prediction_linear[i][0] < 2)):
            counter_1 += 1
        else: counter_0 += 1
print('percentage of errors:',counter/X_test.shape[0]*100, '%')
print('palatability 9 error percentage:', counter_9/counter*100,'%')
print('palatability 8 error percentage:', counter_8/counter*100,'%')
print('palatability 7 error percentage:', counter_7/counter*100,'%')
print('palatability 6 error percentage:', counter_6/counter*100,'%')
print('palatability 5 error percentage:', counter_5/counter*100,'%')
print('palatability 4 error percentage:', counter_4/counter*100,'%')
print('palatability 3 error percentage:', counter_3/counter*100,'%')
print('palatability 2 error percentage:', counter_2/counter*100,'%')
print('palatability 1 error percentage:', counter_1/counter*100,'%')
print('palatability 0 error percentage:', counter_0/counter*100,'%')
