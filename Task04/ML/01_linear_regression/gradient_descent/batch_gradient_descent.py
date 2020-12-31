# batch 批量
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

learn_rate = 0.1  # 超参数
n_iterations = 1000
m = 100

# 初始化theta， w0...wn
theta = np.random.randn(2, 1)
cout = 0
# 求梯度gradient
# gradients = X_b.T.dot(X_b).dot(theta) - X_b.T.dot(y)


# 不会设置阈值，之间设置超参数，迭代次数，迭代次数到了，我们就认为收敛了
for iteration in range(n_iterations):
    cout += 1
    # 调整theta值
    gradients = 1 / m * (X_b.T.dot(X_b).dot(theta) - X_b.T.dot(y))
    theta = theta - gradients * learn_rate

print(cout)
print(theta)


