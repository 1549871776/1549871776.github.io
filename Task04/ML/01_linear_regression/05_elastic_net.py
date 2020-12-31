import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_new = np.array(1.5).reshape(1, -1)

elastic_reg = ElasticNet(alpha=0.0001, l1_ratio=0.15)
elastic_reg.fit(X, y)
y_predict = elastic_reg.predict(X_new)

print(y_predict)
print("w0=", elastic_reg.intercept_)
print("w1=", elastic_reg.coef_)

max_iter = 1000000
n_iter_no_change = 10000
sgd_reg = SGDRegressor(penalty='elasticnet', max_iter=max_iter, n_iter_no_change=n_iter_no_change)
sgd_reg.fit(X, y.ravel())
y_predict_1 = sgd_reg.predict(X_new)
print(y_predict_1)
print("w0=", sgd_reg.intercept_)
print("w1=", sgd_reg.coef_)