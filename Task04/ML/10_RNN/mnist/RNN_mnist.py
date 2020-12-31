import numpy as np
import tensorflow as tf
import gzip
import random

tf.compat.v1.disable_eager_execution()

train_images_file = "MNIST_data_bak/train-images-idx3-ubyte.gz"
train_labels_file = "MNIST_data_bak/train-labels-idx1-ubyte.gz"
t10k_images_file = "MNIST_data_bak/t10k-images-idx3-ubyte.gz"
t10k_labels_file = "MNIST_data_bak/t10k-labels-idx1-ubyte.gz"


def read32(bytestream):
    # 由于网络数据的编码是大端，所以需要加上>
    dt = np.dtype(np.int32).newbyteorder('>')
    data = bytestream.read(4)
    return np.frombuffer(data, dt)[0]


def read_labels(filename):
    with gzip.open(filename) as bytestream:
        magic = read32(bytestream)
        numberOfLabels = read32(bytestream)
        labels = np.frombuffer(bytestream.read(numberOfLabels), np.uint8)
        data = np.zeros((numberOfLabels, 10))
        for i in range(len(labels)):
            data[i][labels[i]] = 1
        bytestream.close()
    return data


def read_images(filename):
    # 把文件解压成字节流
    with gzip.open(filename) as bytestream:
        magic = read32(bytestream)
        numberOfImages = read32(bytestream)
        rows = read32(bytestream)
        columns = read32(bytestream)
        images = np.frombuffer(bytestream.read(numberOfImages * rows * columns), np.uint8)
        images.shape = (numberOfImages, rows * columns)
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        bytestream.close()
    return images


train_labels = read_labels(train_labels_file)
train_images = read_images(train_images_file)
test_labels = read_labels(t10k_labels_file)
test_images = read_images(t10k_images_file)

n_train, n_test, dim, n_classes = train_images.shape[0], test_images.shape[0], train_images.shape[1], train_labels.shape[1]
print(n_train, n_test, dim, n_classes)

dim_input = 28
dim_hidden = 128
dim_output = n_classes
n_steps = 28

weights = {
    'hidden': tf.Variable(tf.compat.v1.random_normal([dim_input, dim_hidden])),
    'out': tf.Variable(tf.compat.v1.random_normal([dim_hidden, dim_output]))
}
biases = {
    'hidden': tf.Variable(tf.compat.v1.random_normal([dim_hidden])),
    'out': tf.Variable(tf.compat.v1.random_normal([dim_output]))
}


def _RNN(_X, _W, _b, _n_steps, _name):
    _X = tf.transpose(_X, [1, 0, 2])
    _X = tf.reshape(_X, [-1, dim_input])
    _H = tf.matmul(_X, _W['hidden']) + _b['hidden']
    _H_split = tf.split(_H, _n_steps, 0)

    lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(dim_hidden, forget_bias=1.0)

    _LSTM_O, _LSTM_S = tf.compat.v1.nn.static_rnn(lstm_cell, _H_split, dtype=tf.float32)

    _O = tf.matmul(_LSTM_O[-1], _W['out']) + _b['out']
    return {
        'X': _X, 'H': _H, 'H_split': _H_split,
        'LSTM_0': _LSTM_O, "_LSTM_S": _LSTM_S, '_O': _O
    }


learning_rate = 0.001
x = tf.compat.v1.placeholder('float', [None, n_steps, dim_input])
y = tf.compat.v1.placeholder('float', [None, dim_output])

my_rnn = _RNN(x, weights, biases, n_steps, 'basic')
pred = my_rnn['_O']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optm = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
init = tf.compat.v1.global_variables_initializer()

training_epochs = 10
batch_size = 16
display_step = 1
sess = tf.compat.v1.Session()
sess.run(init)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_train / batch_size)
    for i in range(100):

        batch_xs = train_images[batch_size * i:batch_size * i + batch_size]
        batch_ys = train_labels[batch_size * i:batch_size * i + batch_size]
        batch_xs = batch_xs.reshape((batch_size, n_steps, dim_input))

        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch

    if epoch % display_step == 0:
        print('Epoch: %03d/%03d cost: %.9f' % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
        print("Training accuracy: %.3f" % (train_acc))

    testimags = test_images.reshape((n_test, n_steps, dim_input))
    test_acc = sess.run(accr, feed_dict={x: testimags, y: test_labels})
    print("Test accuracy: %.3f" % (test_acc))



