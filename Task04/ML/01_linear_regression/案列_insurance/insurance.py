import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./insurance.csv')
# print(type(data_home))
# print(data_home.head())
# print(data_home.tail())
# print(data_home.describe())

# 采样要均匀

data_count = data['age'].value_counts()
# print(data_count)
# data_count.plot(kind='bar')
# plt.show()

# corr -> correlation 相关函数
print(data.corr())

lin_reg = LinearRegression()
x = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = data['charges']

# x['sex'] = x['sex'].map({'male': 0, 'female': 1})
# x['smoker'] = x['smoker'].map({'yes': 0, 'no': 1})

# print(x.region.unique())
# x['region'] = x['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})

x.loc[x['sex'] == 'male', 'sex'] = 0
x.loc[x['sex'] == 'female', 'sex'] = 1

x.loc[x['smoker'] == 'yes', 'smoker'] = 0
x.loc[x['smoker'] == 'no', 'smoker'] = 1

x.loc[x['region'] == 'southwest', 'region'] = 0
x.loc[x['region'] == 'southeast', 'region'] = 1
x.loc[x['region'] == 'northwest', 'region'] = 2
x.loc[x['region'] == 'northeast', 'region'] = 3

ss = StandardScaler()
x_train = ss.fit_transform(x)

ploy_features = PolynomialFeatures(degree=3, include_bias=False)
x_ploy = ploy_features.fit_transform(x_train)
lin_reg.fit(x_ploy, y)

y_predict = lin_reg.predict(x_ploy)

plt.plot(x['age'], y, 'b.')
plt.plot(x['age'], y_predict, 'r.')
plt.show()
