# 机器学习算法（二）:朴素贝叶斯（Naive Bayes）

## 学习地址
https://tianchi.aliyun.com/s/20d6735792ef867814c90698221d1499

## 今天学习的主要内容
朴素贝叶斯是一种分类算法 

## 具体学习内容

### 莺尾花数据集--贝叶斯分类
```
import warnings
warnings.filterwarnings('ignore')
import numpy as np
# 加载莺尾花数据集
from sklearn import datasets
# 导入高斯朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 使用高斯朴素贝叶斯进行计算
clf = GaussianNB(var_smoothing=1e-8)
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
acc = np.sum(y_test == y_pred) / X_test.shape[0]
print("Test Acc : %.3f" % acc)

# 预测
y_proba = clf.predict_proba(X_test[:1])
print(clf.predict(X_test[:1]))
print("预计的概率值:", y_proba)
```
结果：
Test Acc : 0.967  
[2]  
预计的概率值: [[1.63542393e-232 2.18880483e-006 9.99997811e-001]]  


### 模拟离散数据集--贝叶斯分类
```
import random
import numpy as np
# 使用基于类目特征的朴素贝叶斯
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split

# 模拟数据
rng = np.random.RandomState(1)
# 随机生成600个100维的数据，每一维的特征都是[0, 4]之前的整数
X = rng.randint(5, size=(600, 100))

y = np.array([1, 2, 3, 4, 5, 6] * 100)
print(X)
print(y)

data = np.c_[X, y]
print(data)
# X和y进行整体打散
random.shuffle(data)
X = data[:, :-1]
y = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = CategoricalNB(alpha=1)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Test Acc : %.3f" % acc)

# 随机数据测试，分析预测结果，贝叶斯会选择概率最大的预测结果
# 比如这里的预测结果是6，6对应的概率最大，由于我们是随机数据
# 读者运行的时候，可能会出现不一样的结果。
x = rng.randint(5, size=(1, 100))
print(clf.predict_proba(x))
print(clf.predict(x))
```
结果：
Test Acc : 0.683  
[[3.48859652e-04 4.34747491e-04 2.23077189e-03 9.90226387e-01
  5.98248900e-03 7.76745425e-04]]  
[4]  

高斯朴素贝叶斯假设每个特征都服从高斯分布，我们把一个随机变量X服从数学期望为μ，方差为σ^2的数据分布称为高斯分布。对于每个特征我们一般使用平均值来估计μ和使用所有特征的方差估计σ^2。

原理分析：

输入数据为X，输出数据为Y, 根据这两组数据可以很容易得到P(y|x),而当一个新数据到来时，问题转换成已知X，而Y未知，所以遍历所有Y的类型，观察所有的Y与这个X的匹配程度，即计算P(X|Y)，此时采用贝叶斯公式进行转化，故贝叶斯算法由此而来

## 方法记录

### rng = np.random.RandomState(1)函数
numpy.random.RandomState（1）中的1的用法
问题描述：numpy.random.RandomState（1）中1什么用处？

这里的random_state=1指的是伪随机数生成器的种子，简单来说每个种子对应一个随机数或者数列，可以理解为数列识别器。

当seed固定的时候，随机出来的结果相同

参考文档：https://blog.csdn.net/weixin_43415276/article/details/84975464  

### np.r_, np.c_函数
np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等
np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等

参考文档：https://blog.csdn.net/weixin_41797117/article/details/80048688