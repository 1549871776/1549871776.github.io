# 机器学习算法（三）：K近邻(k-nearest neighbors)初探
## KNN的介绍和应用  
- KNN的介绍  
kNN(k-nearest neighbors)，中文翻译K近邻。我们常常听到一个故事：如果要了解一个人的经济水平，只需要知道他最好的5个朋友的经济能力， 对他的这五个人的经济水平求平均就是这个人的经济水平。这句话里面就包含着kNN的算法思想。  

- KNN建立过程
    1. 给定测试样本，计算它与训练集中的每一个样本的距离。
    2. 找出距离近期的K个训练样本。作为测试样本的近邻。
    3. 依据这K个近邻归属的类别来确定样本的类别。

- 类别的判定
    1. 投票决定，少数服从多数。取类别最多的为测试样本类别。
    2. 加权投票法，依据计算得出距离的远近，对近邻的投票进行加权，距离越近则权重越大，设定权重为距离平方的倒数。


## 算法实践

### Demo数据集--kNN分类
```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target


k_list = [1, 3, 5, 8, 10, 15]
h = .02
# 创建不同颜色的画布
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

plt.figure(figsize=(15,14))

# 根据不同的k值进行可视化
for ind,k in enumerate(k_list):
    clf = KNeighborsClassifier(k)
    clf.fit(X, y)
    # 画出决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # 根据边界填充颜色
    Z = Z.reshape(xx.shape)

    plt.subplot(321+ind)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # 数据点可视化到画布
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)"% k)

plt.show()
```

如果选择较小的K值，就相当于用较小的领域中的训练实例进行预测，例如当k=1的时候，在分界点位置的数据很容易受到局部的影响，图中蓝色的部分中还有部分绿色块，主要是数据太局部敏感。当k=15的时候，不同的数据基本根据颜色分开，当时进行预测的时候，会直接落到对应的区域，模型相对更加鲁棒。

### 模拟数据集--kNN回归

```
#Demo来自sklearn官网
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

np.random.seed(0)
# 随机生成40个(0, 1)之前的数，乘以5，再进行升序
X = np.sort(5 * np.random.rand(40, 1), axis=0)
# 创建[0, 5]之间的500个数的等差数列, 作为测试数据
T = np.linspace(0, 5, 500)[:, np.newaxis]
# 使用sin函数得到y值，并拉伸到一维
y = np.sin(X).ravel()
# Add noise to targets[y值增加噪声]
y[::5] += 1 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
# 设置多个k近邻进行比较
n_neighbors = [1, 3, 5, 8, 10, 40]
# 设置图片大小
plt.figure(figsize=(10,20))
for i, k in enumerate(n_neighbors):
    # 默认使用加权平均进行计算predictor
    clf = KNeighborsRegressor(n_neighbors=k, p=2, metric="minkowski")
    # 训练
    clf.fit(X, y)
    # 预测
    y_ = clf.predict(T)
    plt.subplot(6, 1, i + 1)
    plt.scatter(X, y, color='red', label='data')
    plt.plot(T, y_, color='navy', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i)" % (k))

plt.tight_layout()
plt.show()
```
当k=1时，预测的结果只和最近的一个训练样本相关，从预测曲线中可以看出当k很小时候很容易发生过拟合。

当k=40时，预测的结果和最近的40个样本相关，因为我们只有40个样本，此时是所有样本的平均值，此时所有预测值都是均值，很容易发生欠拟合。

一般情况下，使用knn的时候，根据数据规模我们会从[3, 20]之间进行尝试，选择最好的k，例如上图中的[3, 10]相对1和40都是还不错的选择。

### 马绞痛数据--kNN数据预处理+kNN分类pipeline
```
# 下载需要用到的数据集
!wget https://tianchi-media.oss-cn-beijing.aliyuncs.com/DSW/3K/horse-colic.csv

# 下载数据集介绍
!wget https://tianchi-media.oss-cn-beijing.aliyuncs.com/DSW/3K/horse-colic.names

import numpy as np
import pandas as pd
# kNN分类器
from sklearn.neighbors import KNeighborsClassifier
# kNN数据空值填充
from sklearn.impute import KNNImputer
# 计算带有空值的欧式距离
from sklearn.metrics.pairwise import nan_euclidean_distances
# 交叉验证
from sklearn.model_selection import cross_val_score
# KFlod的函数
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2, metric='nan_euclidean')
imputer.fit_transform(X)

nan_euclidean_distances([[np.nan, 6, 5], [3, 4, 3]], [[3, 4, 3], [1, 2, np.nan], [8, 8, 7]])


# load dataset, 将?变成空值
input_file = './horse-colic.csv'
df_data = pd.read_csv(input_file, header=None, na_values='?')

# 得到训练数据和label, 第23列表示是否发生病变, 1: 表示Yes; 2: 表示No. 
data = df_data.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]

# 查看所有特征的缺失值个数和缺失率
for i in range(df_data.shape[1]):
    n_miss = df_data[[i]].isnull().sum()
    perc = n_miss / df_data.shape[0] * 100
    if n_miss.values[0] > 0:
        print('>Feat: %d, Missing: %d, Missing ratio: (%.2f%%)' % (i, n_miss, perc))

# 查看总的空值个数
print('KNNImputer before Missing: %d' % sum(np.isnan(X).flatten()))
# 定义 knnimputer
imputer = KNNImputer()
# 填充数据集中的空值
imputer.fit(X)
# 转换数据集
Xtrans = imputer.transform(X)
# 打印转化后的数据集的空值
print('KNNImputer after Missing: %d' % sum(np.isnan(Xtrans).flatten()))


results = list()
strategies = [str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 18, 20, 21]]
for s in strategies:
    # create the modeling pipeline
    pipe = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=int(s))), ('model', KNeighborsClassifier())])
    # 数据多次随机划分取平均得分
    scores = []
    for k in range(20):
        # 得到训练集合和验证集合, 8: 2
        X_train, X_test, y_train, y_test = train_test_split(Xtrans, y, test_size=0.2)
        pipe.fit(X_train, y_train)
        # 验证model
        score = pipe.score(X_test, y_test)
        scores.append(score)
    # 保存results
    results.append(np.array(scores))
    print('>k: %s, Acc Mean: %.3f, Std: %.3f' % (s, np.mean(scores), np.std(scores)))
# print(results)
# plot model performance for comparison
plt.boxplot(results, labels=strategies, showmeans=True)
plt.show()
```




## 方法记录

### plt.pcolormesh(xx, yy, Z, cmap=cmap_light)函数
- 该函数的作用是绘制分类图
前面两个参数xx，yy为图中的网格点的坐标，Z为每个点坐标的分类，事先用KNN分类算法把每个点分好类，cmap的分类颜色  
参考文档：https://blog.csdn.net/zsdust/article/details/79726118

### 代码: y[::5] += 1 * (0.5 - np.random.rand(8))
- 从第一个值开始，每间5个值增加一个随机值，[::n]表示取间隔值

### plt.boxplot 箱线图
箱线图 又称 ‘ 盒 图 ’  
在1977年由美国的统计学家约翰·图基(John Tukey)发明的。 它由五个数值点组成：最小值(min)，下四分位数(Q1)，中位数(median)，上四分位数(Q3)，最大值(max)。 也可以往盒图里面加入平均值(mean)。下四分位数、中位数、上四分位数组成一个“带有隔间的盒子”。上四分位数到最大值之间建立一条延伸线，这个延伸线成为“胡须(whisker)”。  

```
plt.boxplot(x, notch=None, sym=None, vert=None,   
             whis=None, positions=None, widths=None,   
             patch_artist=None, meanline=None, showmeans=None,   
             showcaps=None, showbox=None, showfliers=None,   
             boxprops=None, labels=None, flierprops=None,   
             medianprops=None, meanprops=None,   
             capprops=None, whiskerprops=None)  

x：指定要绘制箱线图的数据；
notch：是否是凹口的形式展现箱线图，默认非凹口；
sym：指定异常点的形状，默认为+号显示；
vert：是否需要将箱线图垂直摆放，默认垂直摆放；
whis：指定上下须与上下四分位的距离，默认为1.5倍的四分位差；
positions：指定箱线图的位置，默认为[0,1,2…]；
widths：指定箱线图的宽度，默认为0.5；
patch_artist：是否填充箱体的颜色；
meanline：是否用线的形式表示均值，默认用点来表示；
showmeans：是否显示均值，默认不显示；
showcaps：是否显示箱线图顶端和末端的两条线，默认显示；
showbox：是否显示箱线图的箱体，默认显示；
showfliers：是否显示异常值，默认显示；
boxprops：设置箱体的属性，如边框色，填充色等；
labels：为箱线图添加标签，类似于图例的作用；
filerprops：设置异常值的属性，如异常点的形状、大小、填充色等；
medianprops：设置中位数的属性，如线的类型、粗细等；
meanprops：设置均值的属性，如点的大小、颜色等；
capprops：设置箱线图顶端和末端线条的属性，如颜色、粗细等；
whiskerprops：设置须的属性，如颜色、粗细、线的类型等；
```
参考文档：https://blog.csdn.net/weixin_38656890/article/details/80313518