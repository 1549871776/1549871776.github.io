import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# 任何创建的节点会自动加入到默认的图
x1 = tf.Variable(1)
print(x1.graph is tf.compat.v1.get_default_graph())

# 大多数情况下上面运行的很好，有时候或许要管理多个独立的图
# 可以创建一个新的图并且临时使用with块使得它成为默认的如图
graph = tf.Graph()
x3 = tf.Variable(3)
with graph.as_default():
    x2 = tf.Variable(2)

print(x2.graph is graph)
print(x2.graph is tf.compat.v1.get_default_graph())

print(x3.graph is tf.compat.v1.get_default_graph())