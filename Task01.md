# 机器学习算法（一）: 基于逻辑回归的分类预测

## 学习内容概括

### 学习地址

https://dsw-dev.data.aliyun.com/#/

### 今天学习的主要内容   
逻辑回归
- Q：为什么本质是多元线性回归？  
A：1，公式，首先应用了多元线性回归的公式，其次才是把多元线性回归的结果，交给sigmoid函数去进行缩放  
   2，导函数，逻辑回归的损失函数推导的导函数，整个形式上和多元线
性回归基本一致，只是y_hat求解公式包含了一个sigmoid过程而已  

- Q：逻辑回归的损失函数是什么？  
A：交叉熵，做分类就用交叉熵，-y*logP，因为逻辑回归是二分类，所以
loss func = (-y*logP + -(1-y)*log(1-P))，也就是说我们期望这个损失最小然后找到最优解，事实上，我们就可以利用前面学过的梯度下降法来求解最优解了 

- Q：逻辑回归为什么阈值是0.5？  
A：因为线性回归区间是负无穷到正无穷的，所以区间可以按照0来分成两部分，所以带到sigmoid公式里面去z=0的话，y就等于0.5

- Q：逻辑回归做多分类？  
A：逻辑回归做多分类，把多分类的问题，转化成多个二分类的问题，如果假如要分三个类别，就需要同时训练三个互相不影响的模型，比如我们n个维度，那么三分类，w参数的个数就会是(n+1)*3个参数上面所谓的互不影响，指的是模型在梯度下降的时候，分别去训练，分别去下降，三个模型互相不需要传递数据，也不需要等待收敛

## 具体学习内容

### Demo
```
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

##Demo演示LogisticRegression分类

## 构造数据集
x_fearures = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_label = np.array([0, 0, 0, 1, 1, 1])

## 调用逻辑回归模型
lr_clf = LogisticRegression()

## 用逻辑回归模型拟合构造的数据集
lr_clf = lr_clf.fit(x_fearures, y_label) #其拟合方程为 y=w0+w1*x1+w2*x2

## 查看其对应模型的w
print('the weight of Logistic Regression:', lr_clf.coef_)

## 查看其对应模型的w0
print('the intercept(w0) of Logistic Regression:', lr_clf.intercept_)

## 可视化构造的数据样本点
plt.figure()
plt.scatter(x_fearures[:, 0], x_fearures[:, 1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')
plt.show()

# 可视化决策边界
plt.figure()
plt.scatter(x_fearures[:, 0], x_fearures[:, 1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')

nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, nx),np.linspace(y_min, y_max, ny))

z_proba = lr_clf.predict_proba(np.c_[x_grid.ravel(), y_grid.ravel()])
z_proba = z_proba[:, 1].reshape(x_grid.shape)
plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')

plt.show()

### 可视化预测新样本

plt.figure()
## new point 1
x_fearures_new1 = np.array([[0, -1]])
plt.scatter(x_fearures_new1[:,0],x_fearures_new1[:,1], s=50, cmap='viridis')
plt.annotate(s='New point 1',xy=(0,-1),xytext=(-2,0),color='blue',arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='red'))

## new point 2
x_fearures_new2 = np.array([[1, 2]])
plt.scatter(x_fearures_new2[:,0],x_fearures_new2[:,1], s=50, cmap='viridis')
plt.annotate(s='New point 2',xy=(1,2),xytext=(-1.5,2.5),color='red',arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='red'))

## 训练样本
plt.scatter(x_fearures[:,0],x_fearures[:,1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')

# 可视化决策边界
plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')

plt.show()

## 在训练集和测试集上分布利用训练好的模型进行预测
y_label_new1_predict = lr_clf.predict(x_fearures_new1)
y_label_new2_predict = lr_clf.predict(x_fearures_new2)

print('The New point 1 predict class:\n',y_label_new1_predict)
print('The New point 2 predict class:\n',y_label_new2_predict)

## 由于逻辑回归模型是概率预测模型（前文介绍的 p = p(y=1|x,\theta)）,所有我们可以利用 predict_proba 函数预测其概率
y_label_new1_predict_proba = lr_clf.predict_proba(x_fearures_new1)
y_label_new2_predict_proba = lr_clf.predict_proba(x_fearures_new2)

print('The New point 1 predict Probability of each class:\n',y_label_new1_predict_proba)
print('The New point 2 predict Probability of each class:\n',y_label_new2_predict_proba)
```



### 鸢尾花
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from time import time


iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris['DESCR'])
# # print(iris['feature_names'])
X = iris['data_home'][:, 3:]
y = iris['target']


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)  # 取得对应数字在数组中的位置
        for candidate in candidates:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score: {0: .3f} (std: {1: .3f})'.format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print('Parameters: {0}'.format(results['params'][candidate]))
            print('')


start = time()
param_grid = {"tol": [1e-4, 1e-3, 1e-2], "C": [0.4, 0.6, 0.8]}
log_reg = LogisticRegression(multi_class='ovr', solver='sag')  # ovr:二分类问题，sag:通过梯度下降法找到最优解
# log_reg.fit(X, y)
grid_search = GridSearchCV(log_reg, param_grid=param_grid, cv=3)
grid_search.fit(X, y)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

# print(grid_search.cv_results_)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # 在0~3区间段平分1000段，取1000个点
print(X_new)

y_prob = log_reg.predict_proba(X_new)
y_hat = log_reg.predict(X_new)
print(y_prob)
print(y_hat)

plt.plot(X_new, y_prob[:, 2], 'g-', label='Iris-Virginca')
plt.plot(X_new, y_prob[:, 1], 'r-', label='Iris-Versicolour')
plt.plot(X_new, y_prob[:, 0], 'b--', label='Iris-Setosa')
plt.legend()
plt.show()

print(log_reg.predict([[1.7], [1.5]]))
```

## 方法记录
### scatter函数  

matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, *, data=None, **kwargs)

参数的解释：

x，y：表示的是大小为(n,)的数组，也就是我们即将绘制散点图的数据点

s:是一个实数或者是一个数组大小为(n,)，这个是一个可选的参数(改变点的大)。

c:表示的是颜色，也是一个可选项。默认是蓝色'b',表示的是标记的颜色，或者可以是一个表示颜色的字符，或者是一个长度为n的表示颜色的序列等等，感觉还没用到过现在不解释了。但是c不可以是一个单独的RGB数字，也不可以是一个RGBA的序列。可以是他们的2维数组（只有一行）。

marker:表示的是标记的样式，默认的是'o'。

cmap:Colormap实体或者是一个colormap的名字，cmap仅仅当c是一个浮点数数组的时候才使用。如果没有申明就是image.cmap

norm:Normalize实体来将数据亮度转化到0-1之间，也是只有c是一个浮点数的数组的时候才使用。如果没有申明，就是默认为colors.Normalize。

vmin,vmax:实数，当norm存在的时候忽略。用来进行亮度数据的归一化。

alpha：实数，0-1之间。

linewidths:也就是标记点的长度。

参考文档：https://blog.csdn.net/m0_37393514/article/details/81298503

### contour函数
前三个参数 X, Y, Height 用来引进点的位置和对应的高度数据
数组表示选择高度的值
color表示颜色
linewidth表示线段宽度