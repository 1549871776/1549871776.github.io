import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

tf.compat.v1.disable_eager_execution()
# TensorFlow为我们计算梯度，但是同时也给了我们更方便的求解方式
# 它提供给我们与众不同的，有创意的一些优化器，包括梯度下降优化器
# 替换前面代码相应的行，并且一切工作正常

# 设定超参数，Grid_Search进行栅格搜索，其实说白了就是排列组合找到Loss Function最小的时刻
# 的那组超参数结果

n_epochs = 1000
learning_rate = 0.01

# 读取数据，这里读取数据是一下子就把所有的数据交给X， Y节点，所以下面去做梯度下降的时候
# BGD = Batch Gradient Decrease,如果面向数据集比较大的时候，我们倾向于Mini GD

housing = fetch_california_housing(data_home=r"G:\ML\尚学堂\07_tensorfolw_study\data_home", download_if_missing=True)
m, n = housing.data.shape

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# 可以使用TensorFlow或者Numpy或者StandardScaler去进行归一化

scaler = StandardScaler().fit(housing_data_plus_bias)
scaled_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)

# 下面部分X，Y最后用placeholder可以改成Mini BGD
# 构件计算的图
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

# random_uniform函数创建图里一个节点包含随机数值，给定它的形状和取值范围，就像numpy里的rand（）

theta = tf.Variable(tf.compat.v1.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')

# 梯度的公式：(y_pred - y) * xj
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
# gradients = tf.gradients(mse, [theta])[0]
# # 赋值函数对于BGD来说就是 theta_new = theta - (learning_rate * gradients)
# training_op = tf.compat.v1.assign(theta, theta - learning_rate * gradients)

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
# MomentumpOptimizer收敛回避梯度下降更快
# optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)

init = tf.compat.v1.global_variables_initializer()

# 下面开始训练
with tf.compat.v1.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch:", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    print(best_theta)