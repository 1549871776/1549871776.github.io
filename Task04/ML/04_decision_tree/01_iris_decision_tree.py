import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl

iris = load_iris()
data = pd.DataFrame(iris.data)
data.columns = iris.feature_names
data['Species'] = load_iris().target
# print(data_home)

x = data.iloc[:, :2]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=42)
tree_clf = DecisionTreeClassifier(max_depth=8, criterion='entropy')
tree_clf.fit(x_train, y_train)
y_test_hat = tree_clf.predict(x_test)

print("acc score:", accuracy_score(y_test, y_test_hat))

depth = range(1, 15)
err_list = []
for i in depth:
    tree_clf = DecisionTreeClassifier(max_depth=i, criterion='entropy')
    tree_clf.fit(x_train, y_train)
    y_test_hat = tree_clf.predict(x_test)
    result = (y_test_hat == y_test)
    error = 1 - np.mean(result)
    err_list.append(error)
    print(i, '错误率: %2.f%%' % (100 * error))

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
plt.figure(facecolor='w')  # 图的底色是白色
plt.plot(depth, err_list, 'ro-', lw=2)  # red 点以o型圈相连 线的宽度为2
plt.xlabel('决策树深度', fontsize=15)
plt.ylabel('错误率', fontsize=15)
plt.title('决策树深度和过拟合', fontsize=18)
plt.grid(True)
plt.show()
