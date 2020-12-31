import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


# 前面的代码执行的不错，但是它需要数学上通过损失函数MSE来求导梯度
# 在线性回归的例子中，这样是可以的，看起来通过数学公式去求解不难
# 但如果是深度学习，我们很难这样去做，会比较头疼，会很容易出错
# 幸运的是，TensorFlow提供的autodiff特性可以自动的并有效的计算梯度为我们
# reverse-mode autodiff
tf.compat.v1.disable_eager_execution()

n_epochs = 1000
learning_rate = 0.01

housing = fetch_california_housing(data_home=r"G:\ML\尚学堂\07_tensorfolw_study\data_home", download_if_missing=True)
m, n = housing.data.shape

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# 可以使用TensorFlow或者Numpy或者sklearn的StandardScaler去进行归一化
scaler = StandardScaler().fit(housing_data_plus_bias)
scaled_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

# random_uniform函数创建图里一个节点包含随机数值，给定他的形状和取值范围，就像numpy里的rand()
theta = tf.Variable(tf.compat.v1.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
# 梯度的公式：(y_pred - y) * xj
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
gradients = tf.gradients(mse, [theta])[0]
# 赋值函数对于BGD来说就是 theta_new = theta - (learning_rate * gradients)
training_op = tf.compat.v1.assign(theta, theta - learning_rate * gradients)


init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch:', epoch, 'MSE=', mse.eval())
        sess.run(training_op)
    theta_best = theta.eval()
    print(theta_best)

