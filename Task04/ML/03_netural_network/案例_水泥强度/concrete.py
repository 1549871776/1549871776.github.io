import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

data = pd.read_csv(r'concrete.csv')

print(data.columns.values)
datalist = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']

# def normalize(data_home):
#     data_home = (data_home - data_home.min()) / (data_home.max() - data_home.min())
#     return data_home


def normalize(data):
    for item in datalist:
        data[item] = (data[item] - data[item].min()) / (data[item].max() - data[item].min())
    return data


x = normalize(data)
# x = data_home[datalist]
y = data['strength']


print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = MLPRegressor(max_iter=20000)


clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(y_predict.shape)
print(y_predict)
print(y_test.shape)
plt.plot(range(0, 309), y_predict, 'r.')
plt.plot(range(0, 309), y_test, 'b*')
plt.show()
