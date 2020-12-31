import tensorflow as tf
import numpy as np
import gzip
import random
###################################################################################

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

###################################################################################


max_steps = 1000
learning_rate = 0.001
dropout = 0.9

log_dir = './logs/mnist_with_summaries'

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.InteractiveSession()

with tf.name_scope('input'):
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 784), name='x-input')
    y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name='y-input')

with tf.name_scope('input_reshape'):
    # 784维度变形为图片保持到节点
    # -1 代表进来图片的数量，28，29是图片的高和宽，1是图片的颜色通道
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.compat.v1.summary.image('input', image_shaped_input, 10)


# 定义神经网络的初始化方法
def weight_variable(shape):
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
        tf.compat.v1.summary.histogram('histogram', var)


# 设计一个MLP多层神经网络来训练数据
# 在每一层中都对应模型进行汇总
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.compat.v1.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.compat.v1.summary.histogram('activations', activations)
        return activations
    

# 我们使用刚刚定义的函数创建一层神经网络，输入维度是图片的尺寸784=28*28
# 输出的维度是隐藏节点数500，再创建一个Dropout层，并使用tf.summary.scaler记录keep_prob
# 然后使用nn_layer定义神经网络输出层，其输入维度为上一层隐含节点数500，输出维度为类别数10
# 同时激活函数为全等映射identity，暂时不使用softmax
hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    tf.compat.v1.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.compat.v1.nn.dropout(hidden1, keep_prob)


y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)


# 使用tf.nn.softmax_cross_entropy_with_logits()对前面的输出层的结果进行Softmax
# 处理并计算交叉熵损失cross_entropy,计算平均的损失，使用tf.summary.scalar进行统计汇总
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)

# 下面使用Adam优化器对损失进行优化，同时统计预测正确的样本数并计算正确率accuracy，汇总
with tf.name_scope('train'):
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.compat.v1.summary.scalar('accuracy', accuracy)

# 因为我们之前定义了太多的tf.summary汇总操作。逐一执行这些操作太麻烦
# 使用tf.summary.merge_all()直接获取所有汇总操作，以便后面执行
merged = tf.compat.v1.summary.merge_all()
# 定义两个tf.summary.FileWriter文件记录器在不同的子目录，分别用来存储训练和测试的日志数据
train_writer = tf.compat.v1.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.compat.v1.summary.FileWriter(log_dir + '/test')
# 同时，将Session计算图sess.graphy加入训练过程，这样再TensorBoard的GRAPHS窗口中就能展示
# 整个计算图的可视化效果，最后初始化全部变量
tf.compat.v1.global_variables_initializer().run()


# 定义feed_dict函数，如果是训练，需要设置dropout，如果是测试，keep_prob设置为1
def feed_dict(train):
    if train:
        rd = random.randint(0, 550)
        xs = train_images[100 * rd:100*rd + 100]
        ys = train_labels[100 * rd:100*rd + 100]
        k = dropout
    else:
        xs = test_images
        ys = test_labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}


# 执行训练，测试，日志记录操作
# 创建模型的保存器
saver = tf.compat.v1.train.Saver()
for i in range(max_steps):
    if i%10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:
        if i%100 == 99:
            run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, 1)
            saver.save(sess, log_dir + 'model.ckpt', i)
            print('Adding run metadata for', i)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()


