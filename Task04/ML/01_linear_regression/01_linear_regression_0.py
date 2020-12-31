import numpy as np
import matplotlib.pyplot as plt

# 这里相当于是随机X维度1，rand是随机均匀分布
X = 2 * np.random.rand(100, 1)
# 人为设置真实的Y一列，np.random.randn(100,1)是设置error，randn是标准正态分布
y = 4 + 3 * X + np.random.randn(100, 1)
# 整合X0和X1, c_理解为conbine
X_b = np.c_[np.ones((100, 1)), X]
print(X_b)

# 常规等式求解theta，inv求逆，dot点积
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)

# 创建测试集里的X1
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
print(X_new_b)

y_predict = X_new_b.dot(theta_best)
print('预测值为：')
print(y_predict)

plt.plot(X_new, y_predict, 'r-')  # 将预测的两个点用直线相连
plt.plot(X, y, 'b.')  # 将数据以点阵显示
plt.axis([0, 2, 0, 15])
plt.show()