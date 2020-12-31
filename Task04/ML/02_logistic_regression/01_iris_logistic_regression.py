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



