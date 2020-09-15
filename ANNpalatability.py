import pandas as pd
from keras.engine.saving import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

df = pd.read_csv('Datasets/balanced_W_350K.csv')

'''
    reversing the palatability value in 
    order to have higher values for better food baskets
'''
max_palatability = max(df['lable'].max(), df['lable'].max())
df['lable'] = max_palatability - df['lable']
dataset = df.copy()

features_name = ['Beans', 'Bulgur', 'Cheese', 'Fish', 'Meat', 'Corn-soya blend (CSB)', 'Dates',
                 'Dried skim milk (enriched) (DSM)', 'Milk', 'Salt', 'Lentils', 'Maize', 'Maize meal', 'Chickpeas',
                 'Rice',
                 'Sorghum/millet', 'Soya-fortified bulgur wheat', 'Soya-fortified maize meal',
                 'Soya-fortified sorghum grits', 'Soya-fortified wheat flour', 'Sugar', 'Oil', 'Wheat', 'Wheat flour',
                 'Wheat-soya blend (WSB)']

target_name = ['lable']

# Scaling the labels
sc_t = MinMaxScaler(feature_range=(0, 1))
dataset[target_name] = sc_t.fit_transform((dataset[target_name]))

X = dataset[features_name]
y = dataset[target_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = load_model('Gmodels/simple2.h5')  # loading of a pre trained model

### ANN REGRESSION ###
# model = Sequential()
# model.add(Dense(units=5, input_dim=len(features_name), activation='tanh', use_bias=True))
# model.add(Dense(units=2, activation='tanh', use_bias=True))
# model.add(Dense(units=1, activation='sigmoid', use_bias=True))
# model.compile(loss='mse', optimizer='Adam', metrics=['mse', 'mae'])
# opt = keras.optimizers.Adam(lr=0.0001)
# model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
# model.fit(X_train, y_train, epochs=5000, batch_size=32, verbose=2)
# model.save('modelsOL/simple_datasetOL.h5')  # Saving the model


'''
    Test
'''
prediction_ANN = model.predict(X_test) * 10
y_test = y_test * 10

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
counter_85 = 0
counter_95= 0
counter_75 = 0
counter_65 = 0
counter_55 = 0
counter_45 = 0
counter_35 = 0
counter_25 = 0
counter_15 = 0
counter_05 = 0

for i in range(0, len(prediction_ANN)):
    if abs((y_test[i:i + 1]['lable'] - prediction_ANN[i][0]).values[0]) >= 0.7:
        counter += 1
        if((prediction_ANN[i][0] >= 9.5) & (prediction_ANN[i][0] < 10)):
            counter_95 += 1
        elif((prediction_ANN[i][0] >= 9) & (prediction_ANN[i][0] < 9.5)):
            counter_9 += 1
        elif((prediction_ANN[i][0] >= 8.5) & (prediction_ANN[i][0] < 9)):
            counter_85 += 1
        elif((prediction_ANN[i][0] >= 8) & (prediction_ANN[i][0] < 8.5)):
            counter_8 += 1
        elif((prediction_ANN[i][0] >= 7.5) & (prediction_ANN[i][0] < 8)):
            counter_75 += 1
        elif((prediction_ANN[i][0] >= 7) & (prediction_ANN[i][0] < 7.5)):
            counter_7 += 1
        elif((prediction_ANN[i][0] >= 6.5) & (prediction_ANN[i][0] < 7)):
            counter_65 += 1
        elif((prediction_ANN[i][0] >= 6) & (prediction_ANN[i][0] < 6.5)):
            counter_6 += 1
        elif((prediction_ANN[i][0] >= 5.5) & (prediction_ANN[i][0] < 6)):
            counter_55 += 1
        elif((prediction_ANN[i][0] >= 5) & (prediction_ANN[i][0] < 5.5)):
            counter_5 += 1
        elif((prediction_ANN[i][0] >= 4.5) & (prediction_ANN[i][0] < 5)):
            counter_45 += 1
        elif((prediction_ANN[i][0] >= 4) & (prediction_ANN[i][0] < 4.5)):
            counter_4 += 1
        elif((prediction_ANN[i][0] >= 3.5) & (prediction_ANN[i][0] < 4)):
            counter_35 += 1
        elif((prediction_ANN[i][0] >= 3) & (prediction_ANN[i][0] < 3.5)):
            counter_3 += 1
        elif((prediction_ANN[i][0] >= 2.5) & (prediction_ANN[i][0] < 3)):
            counter_25 += 1
        elif((prediction_ANN[i][0] >= 2) & (prediction_ANN[i][0] < 2.5)):
            counter_2 += 1
        elif((prediction_ANN[i][0] >= 1.5) & (prediction_ANN[i][0] < 2)):
            counter_15 += 1
        elif ((prediction_ANN[i][0] >= 1) & (prediction_ANN[i][0] < 1.5)):
            counter_1 += 1
        elif ((prediction_ANN[i][0] >= 0.5) & (prediction_ANN[i][0] < 1)):
            counter_05 += 1
        else: counter_0 += 1
print('percentage of errors:',counter/X_test.shape[0]*100, '%')
print(counter_95/counter*100,'%')
print(counter_9/counter*100,'%')
print(counter_85/counter*100,'%')
print(counter_8/counter*100,'%')
print(counter_75/counter*100,'%')
print(counter_7/counter*100,'%')
print(counter_65/counter*100,'%')
print(counter_6/counter*100,'%')
print(counter_55/counter*100,'%')
print(counter_5/counter*100,'%')
print(counter_45/counter*100,'%')
print(counter_4/counter*100,'%')
print(counter_35/counter*100,'%')
print(counter_3/counter*100,'%')
print(counter_25/counter*100,'%')
print(counter_2/counter*100,'%')
print(counter_15/counter*100,'%')
print(counter_1/counter*100,'%')
print(counter_05/counter*100,'%')
print(counter_0/counter*100,'%')