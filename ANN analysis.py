import math
from keras.engine.saving import load_model
import numpy as np
import pandas as pd

file_ANN_simple = 'Gmodels/simple2.h5'
ANN = load_model(file_ANN_simple)



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

# for i in range(0, len(prediction_ANN)):
#     if abs((y_test[i:i + 1]['lable'] - prediction_ANN[i][0]).values[0]) >= 0.5:
#         counter += 1
#         if((prediction_ANN[i][0] >= 9.5) & (prediction_ANN[i][0] < 10)):
#             counter_95 += 1
#         elif((prediction_ANN[i][0] >= 9) & (prediction_ANN[i][0] < 9.5)):
#             counter_9 += 1
#         elif((prediction_ANN[i][0] >= 8.5) & (prediction_ANN[i][0] < 9)):
#             counter_85 += 1
#         elif((prediction_ANN[i][0] >= 8) & (prediction_ANN[i][0] < 8.5)):
#             counter_8 += 1
#         elif((prediction_ANN[i][0] >= 7.5) & (prediction_ANN[i][0] < 8)):
#             counter_75 += 1
#         elif((prediction_ANN[i][0] >= 7) & (prediction_ANN[i][0] < 7.5)):
#             counter_7 += 1
#         elif((prediction_ANN[i][0] >= 6.5) & (prediction_ANN[i][0] < 7)):
#             counter_65 += 1
#         elif((prediction_ANN[i][0] >= 6) & (prediction_ANN[i][0] < 6.5)):
#             counter_6 += 1
#         elif((prediction_ANN[i][0] >= 5.5) & (prediction_ANN[i][0] < 6)):
#             counter_55 += 1
#         elif((prediction_ANN[i][0] >= 5) & (prediction_ANN[i][0] < 5.5)):
#             counter_5 += 1
#         elif((prediction_ANN[i][0] >= 4.5) & (prediction_ANN[i][0] < 5)):
#             counter_45 += 1
#         elif((prediction_ANN[i][0] >= 4) & (prediction_ANN[i][0] < 4.5)):
#             counter_4 += 1
#         elif((prediction_ANN[i][0] >= 3.5) & (prediction_ANN[i][0] < 4)):
#             counter_35 += 1
#         elif((prediction_ANN[i][0] >= 3) & (prediction_ANN[i][0] < 3.5)):
#             counter_3 += 1
#         elif((prediction_ANN[i][0] >= 2.5) & (prediction_ANN[i][0] < 3)):
#             counter_25 += 1
#         elif((prediction_ANN[i][0] >= 2) & (prediction_ANN[i][0] < 2.5)):
#             counter_2 += 1
#         elif((prediction_ANN[i][0] >= 1.5) & (prediction_ANN[i][0] < 2)):
#             counter_15 += 1
#         elif ((prediction_ANN[i][0] >= 1) & (prediction_ANN[i][0] < 1.5)):
#             counter_1 += 1
#         elif ((prediction_ANN[i][0] >= 0.5) & (prediction_ANN[i][0] < 1)):
#             counter_05 += 1
#         else: counter_0 += 1
# print('percentage of errors:',counter/X_test.shape[0]*100, '%')
# print(counter_95/counter*100,'%')
# print(counter_9/counter*100,'%')
# print(counter_85/counter*100,'%')
# print(counter_8/counter*100,'%')
# print(counter_75/counter*100,'%')
# print(counter_7/counter*100,'%')
# print(counter_65/counter*100,'%')
# print(counter_6/counter*100,'%')
# print(counter_55/counter*100,'%')
# print(counter_5/counter*100,'%')
# print(counter_45/counter*100,'%')
# print(counter_4/counter*100,'%')
# print(counter_35/counter*100,'%')
# print(counter_3/counter*100,'%')
# print(counter_25/counter*100,'%')
# print(counter_2/counter*100,'%')
# print(counter_15/counter*100,'%')
# print(counter_1/counter*100,'%')
# print(counter_05/counter*100,'%')
# print(counter_0/counter*100,'%')

solution = [0.13, 0, 0.102, 0,
            0, 0, 0.109, 0.198,
            0, 0.05, 0, 0,
            0, 0.691, 0, 3.798,
            0, 0, 0, 0,
            0, 0.385, 1.102, 0, 0.37]

G1 = [1, 11, 12, 14, 15, 22, 23]
G2 = [0, 6, 10, 13]
G3 = [21]
G4 = [5, 16, 17, 18, 19, 24]
G5 = [2, 3, 4, 7, 8]

Sg1 = (sum(solution[i] for i in G1) * 100 - 375)
Sg2 = (sum(solution[i] for i in G2) * 100 - 80)
Sg3 = (sum(solution[i] for i in G3) * 100 - 27.5)
Sg4 = (sum(solution[i] for i in G4) * 100 - 30)
Sg5 = (sum(solution[i] for i in G5) * 100 - 20)

print(Sg1, Sg2, Sg3, Sg4, Sg5)
max_palatability = 279.96
max_palatability_nonW = 139.69
weighted_palatability = np.round(math.sqrt(Sg1 ** 2 + (2.5 * Sg2) ** 2
                                           + (10 * Sg3) ** 2 + (4.2 * Sg4) ** 2
                                           + (6.25 * Sg5) ** 2), 2)
nonweighted_palatability = np.round(math.sqrt(Sg1 ** 2 + (Sg2) ** 2
                                              + (Sg3) ** 2 + (Sg4) ** 2
                                              + (Sg5) ** 2), 2)
real_palatability = max(max_palatability - weighted_palatability, 0)
print('real palatability:', real_palatability)
print('predicted palatability:', ANN.predict(np.array([solution]))[0][0]*10)


