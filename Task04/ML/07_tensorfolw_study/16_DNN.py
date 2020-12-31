import tensorflow as tf
import numpy as np
import gzip
tf.compat.v1.disable_eager_execution()


# 构件图阶段
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.compat.v1.placeholder(tf.int64, shape=(None, 10), name='y')

######################################################################################
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

######################################################################################


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.compat.v1.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        w = tf.Variable(init, name='weight')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        z = tf.matmul(X, w) + b
        if activation == 'relu':
            return tf.nn.relu(z)
        else:
            return z


# with tf.name_scope('dnn'):
#     hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
#     hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
#     logits = neuron_layer(hidden2, n_outputs, "outputs")

with tf.name_scope('dnn'):
    hidden1 = tf.keras.layers.Dense(n_hidden1, name='hidden1', activation='relu')(X)
    hidden2 = tf.keras.layers.Dense(n_hidden2, name='hidden2', activation='relu')(hidden1)
    logits = tf.keras.layers.Dense(n_outputs, name='outputs')(hidden2)


with tf.name_scope('loss'):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.compat.v1.global_variables_initializer()

n_epochs = 40
batch_size = 50

with tf.compat.v1.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(55000 // batch_size):
            batch_xs = train_images[batch_size * iteration:batch_size * iteration + batch_size]
            batch_ys = train_labels[batch_size * iteration:batch_size * iteration + batch_size]
            sess.run(training_op, feed_dict={X: batch_xs, y: batch_ys})
        acc_train = accuracy.eval(feed_dict={X: batch_xs, y: batch_ys})
        acc_test = accuracy.eval({X: test_images, y: test_labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)





