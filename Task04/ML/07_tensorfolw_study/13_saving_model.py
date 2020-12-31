
# 有时候需要把模型保存起来，有时候需要做一些checkpoint在训练中
# 以至于如果计算机宕机，我们还可以从之前的checkpoint的位置去继续
# TensorFlow使得我们去保存和加载模型非常方便，仅需要去创建Saver节点在构建阶段最后
# 然后在计算阶段去调用save()


import tensorflow as tf
import gzip
import sys
import struct
import numpy


train_images_file = "MNIST_data_bak/train-images-idx3-ubyte.gz"
train_labels_file = "MNIST_data_bak/train-labels-idx1-ubyte.gz"
t10k_images_file = "MNIST_data_bak/t10k-images-idx3-ubyte.gz"
t10k_labels_file = "MNIST_data_bak/t10k-labels-idx1-ubyte.gz"


def read32(bytestream):
    # 由于网络数据的编码是大端，所以需要加上>
    dt = numpy.dtype(numpy.int32).newbyteorder('>')
    data = bytestream.read(4)
    return numpy.frombuffer(data, dt)[0]


def read_labels(filename):
    with gzip.open(filename) as bytestream:
        magic = read32(bytestream)
        numberOfLabels = read32(bytestream)
        labels = numpy.frombuffer(bytestream.read(numberOfLabels), numpy.uint8)
        data = numpy.zeros((numberOfLabels, 10))
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
        images = numpy.frombuffer(bytestream.read(numberOfImages * rows * columns), numpy.uint8)
        images.shape = (numberOfImages, rows * columns)
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
        bytestream.close()
    return images


train_labels = read_labels(train_labels_file)
train_images = read_images(train_images_file)
test_labels = read_labels(t10k_labels_file)
test_images = read_images(t10k_images_file)


tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 784))

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)

# 训练
# labels是每张图片都对应一个one-hot的10个值的向量
y_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 10))
# 定义损失函数，交叉熵损失函数
# 对于多分类问题，通常使用交叉熵损失函数
# axis,指明按照每行加，还是按照每列加
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.compat.v1.log(y), axis=1))

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# 初始化变量
init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()

n_epoch = 500
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for num in range(2):
        for epoch in range(n_epoch):
            if (epoch + (num * 500)) % 100 == 0:
                print("第 %s 次" % (epoch + (num * 500)))
                save_path = saver.save(sess, "./ckpt/my_model.ckpt")

            batch_xs = train_images[100 * epoch:100 * epoch + 100]
            batch_ys = train_labels[100 * epoch:100 * epoch + 100]
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    best_theta = w.eval()
    save_path = saver.save(sess, "./ckpt/my_model_final.ckpt")






