import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, 'b.')

d = {1: 'g-', 2: 'r+', 10: 'y*'}

# 遍历字典，返回键名
for i in d:
    poly_features = PolynomialFeatures(degree=i, include_bias=False)  # degree表示转换的维度
    X_poly = poly_features.fit_transform(X)
    lin_reg = LinearRegression()  # 这里设置了w0
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)

    y_predict = lin_reg.predict(X_poly)
    plt.plot(X_poly[:, 0], y_predict, d[i])
plt.show()
