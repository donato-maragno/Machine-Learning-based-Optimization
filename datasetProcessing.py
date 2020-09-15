import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Datasets/datasetC.csv')
target = ['lable']
sc_t = MinMaxScaler(feature_range=(0, 10))
df[target] = sc_t.fit_transform((df[target]))
x0 = df.loc[(df['lable'] < 1)]
x1 = df.loc[(df['lable'] > 1) & (df['lable'] <= 2)]
x2 = df.loc[(df['lable'] > 2) & (df['lable'] <= 3)]
x3 = df.loc[(df['lable'] > 3) & (df['lable'] <= 4)]
# plt.hist([x0['lable'], x1['lable'], x2['lable'], x3['lable']], color=['#E69F00', '#ff7f0e', '#6495ED', '#191970'], bins=1000)
plt.hist(df['lable'], bins=1000)
plt.xticks(np.arange(0.0, 10.05, 0.5))
plt.legend()
plt.show()

# G1 = [1, 11, 12, 14, 15, 22, 23]
# G2 = [0, 6, 10, 13]
# G3 = [21]
# G4 = [5, 16, 17, 18, 19, 24]
# G5 = [2, 3, 4, 7, 8]
# df['n_neg_Sg'] = 0
#
# for i in range(0, df.shape[0]):
#     solution = df[i:i+1]
#     solution = solution.rename(columns={x:y for x,y in zip(solution.columns, range(0,len(solution.columns)))})
#     Sg =[]
#     Sg.append(sum(solution[j] for j in G1) * 100 - 375)
#     Sg.append(sum(solution[j] for j in G2) * 100 - 80)
#     Sg.append(sum(solution[j] for j in G3) * 100 - 27.5)
#     Sg.append(sum(solution[j] for j in G4) * 100 - 30)
#     Sg.append(sum(solution[j] for j in G5) * 100 - 20)
#     for value in Sg:
#         if(value.values<0):
#             print(i)
#             df['n_neg_Sg'][i] += 1



def dataset_balanced(dataset):
    df = dataset
    max_palatability = 279.96

    features_name = ['Beans', 'Bulgur', 'Cheese', 'Fish', 'Meat', 'Corn-soya blend (CSB)', 'Dates',
                     'Dried skim milk (enriched) (DSM)', 'Milk', 'Salt', 'Lentils', 'Maize', 'Maize meal', 'Chickpeas',
                     'Rice',
                     'Sorghum/millet', 'Soya-fortified bulgur wheat', 'Soya-fortified maize meal',
                     'Soya-fortified sorghum grits', 'Soya-fortified wheat flour', 'Sugar', 'Oil', 'Wheat', 'Wheat flour',
                     'Wheat-soya blend (WSB)']

    dataset9_5 = (df.loc[df['lable'] >= max_palatability / 10 * 9.5]).sample(frac=1)[0:2000]
    dataset9 = (df.loc[(df['lable'] >= max_palatability / 10 * 9) & (df['lable'] < max_palatability / 10 * 9.5)]).sample(
        frac=1)[0:2000]
    dataset8_5 = (df.loc[(df['lable'] >= max_palatability / 10 * 8.5) & (df['lable'] < max_palatability / 10 * 9)]).sample(
        frac=1)[0:2000]
    dataset8 = (df.loc[(df['lable'] >= max_palatability / 10 * 8) & (df['lable'] < max_palatability / 10 * 8.5)]).sample(
        frac=1)[0:2000]
    dataset7_5 = (df.loc[(df['lable'] >= max_palatability / 10 * 7.5) & (df['lable'] < max_palatability / 10 * 8)]).sample(
        frac=1)[0:2000]
    dataset7 = (df.loc[(df['lable'] >= max_palatability / 10 * 7) & (df['lable'] < max_palatability / 10 * 7.5)]).sample(
        frac=1)[0:2000]
    dataset6_5 = (df.loc[(df['lable'] >= max_palatability / 10 * 6.5) & (df['lable'] < max_palatability / 10 * 7)]).sample(
        frac=1)[0:2000]
    dataset6 = (df.loc[(df['lable'] >= max_palatability / 10 * 6) & (df['lable'] < max_palatability / 10 * 6.5)]).sample(
        frac=1)[0:2000]
    dataset5_5 = (df.loc[(df['lable'] >= max_palatability / 10 * 5.5) & (df['lable'] < max_palatability / 10 * 6)]).sample(
        frac=1)[0:2000]
    dataset5 = (df.loc[(df['lable'] >= max_palatability / 10 * 5) & (df['lable'] < max_palatability / 10 * 5.5)]).sample(
        frac=1)[0:2000]
    dataset4_5 = (df.loc[(df['lable'] >= max_palatability / 10 * 4.5) & (df['lable'] < max_palatability / 10 * 5)]).sample(
        frac=1)[0:2000]
    dataset4 = (df.loc[(df['lable'] >= max_palatability / 10 * 4) & (df['lable'] < max_palatability / 10 * 4.5)]).sample(
        frac=1)[0:2000]
    dataset3_5 = (df.loc[(df['lable'] >= max_palatability / 10 * 3.5) & (df['lable'] < max_palatability / 10 * 4)]).sample(
        frac=1)[0:2000]
    dataset3 = (df.loc[(df['lable'] >= max_palatability / 10 * 3) & (df['lable'] < max_palatability / 10 * 3.5)]).sample(
        frac=1)[0:2000]
    dataset2_5 = (df.loc[(df['lable'] >= max_palatability / 10 * 2.5) & (df['lable'] < max_palatability / 10 * 3)]).sample(
        frac=1)[0:2000]
    dataset2 = (df.loc[(df['lable'] >= max_palatability / 10 * 2) & (df['lable'] < max_palatability / 10 * 2.5)]).sample(
        frac=1)[0:2000]
    dataset1_5 = (df.loc[(df['lable'] >= max_palatability / 10 * 1.5) & (df['lable'] < max_palatability / 10 * 2)]).sample(
        frac=1)[0:2000]
    dataset1 = (df.loc[(df['lable'] >= max_palatability / 10 * 1) & (df['lable'] < max_palatability / 10 * 1.5)]).sample(
        frac=1)[0:2000]
    dataset0_5 = (df.loc[(df['lable'] >= max_palatability / 10 * 0.5) & (df['lable'] < max_palatability / 10 * 1)]).sample(
        frac=1)[0:2000]
    dataset0 = (df.loc[df['lable'] < max_palatability / 10 * 0.5]).sample(frac=1)[0:2000]
    min = df.loc[df['lable'] == df['lable'].min()][0:1]
    max = df.loc[df['lable'] == df['lable'].max()][0:1]

    dataset = pd.concat(
        [dataset0_5, dataset1_5, dataset2_5, dataset3_5, dataset4_5, dataset5_5, dataset6_5, dataset7_5, dataset8_5, dataset9_5,
         dataset0, dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, min, max])
    # dataset.drop('lable', 1, inplace=True)
    dataset.to_csv(index=False, path_or_buf='Datasets/datasetOL.csv')

