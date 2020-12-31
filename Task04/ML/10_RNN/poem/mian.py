#! /usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import numpy as np
import tensorflow as tf
import io
import os
tf.compat.v1.disable_eager_execution()

# -------------------------------数据预处理---------------------------#
poetry_file = 'poetry.txt'

# 诗集
poetrys = []
with io.open(poetry_file, "r", encoding='utf-8') as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if '_' in content or '(' in content or ')' in content or '《' in content or '[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = '[' + content + ']'
            poetrys.append(content)
        except Exception as e:
            pass

# 按诗的字数排序
poetrys = sorted(poetrys, key=lambda line: len(line))


# 统计每个字出现的次数
all_words = []
for poetry in poetrys:
    all_words += [word for word in poetry]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)

# 取前多少个字
words = words[:len(words)] + (' ',)

# 每个字映射一个ID
word_num_map = dict(zip(words, range(len(words))))
# 把诗转换成向量形式
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]

# 每次取64首诗进行训练
batch_size = 64
n_chunk = len(poetrys_vector) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size

    batches = poetrys_vector[start_index: end_index]
    length = max(map(len, batches))
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:, : -1] = xdata[:, 1:]
    x_batches.append(xdata)
    y_batches.append(ydata)


# ---------------------------------------10_RNN-------------------------------------- #

input_data = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
output_targets = tf.compat.v1.placeholder(tf.int32, [batch_size, None])


# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
    if model == 'rnn':
        cell_fun = tf.compat.v1.nn.rnn_cell.BasicRNNCell
    elif model == 'gtu':
        cell_fun = tf.compat.v1.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fun = tf.compat.v1.nn.rnn_cell.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.compat.v1.variable_scope('rnnlm'):
        softmax_w = tf.compat.v1.get_variable("sotfmax_w", [rnn_size, len(words) + 1])
        softmax_b = tf.compat.v1.get_variable("softmax_b", [len(words) + 1])

        with tf.device('/cpu:0'):
            embedding = tf.compat.v1.get_variable("embedding", [len(words) + 1, rnn_size])
            inputs = tf.compat.v1.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = tf.compat.v1.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs, [-1, rnn_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    return logits, last_state, probs, cell, initial_state


ckpt_dir = "ckpt_dir"
if not os.path.exists(ckpt_dir):
  os.makedirs(ckpt_dir)


# 训练
def train_neural_network():
    logits, last_state, probability, _, _ = neural_network()
    labels = tf.one_hot(tf.reshape(output_targets, [-1]), depth=len(words) + 1)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(loss)
    learn_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.compat.v1.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.compat.v1.train.AdamOptimizer(learn_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.initialize_all_variables())
        saver = tf.compat.v1.train.Saver(tf.compat.v1.all_variables())

        for epoch in range(295):
            sess.run(tf.compat.v1.assign(learn_rate, 0.002 * (0.97 ** epoch)))
            n = 0
            for batche in range(n_chunk):
                train_loss, _, _, accuracy = sess.run([cost, last_state, train_op, accr], feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})
                n += 1
                print('Epoch: %03d batch:%03d cost: %.9f Accuracy: %.3f %%' % (epoch, batche, train_loss, accuracy * 100))
            if epoch % 7 == 0:
                saver.save(sess, ckpt_dir + '/poetry.module', global_step=epoch)


train_neural_network()
