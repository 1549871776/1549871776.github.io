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

saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    saver.restore(sess, "./ckpt/my_model_final.ckpt")
    # 评估
    # tf.argmax()是从tensor中寻找最大值的序号，tf.argmax就是求各个预测的数字中概率最大的那一个
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # 用tf.cast将之前的correct_prediction输出的bool值转换成float32，在求平均
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("TestSet acc: %s" % accuracy.eval({x: test_images, y_: test_labels}))