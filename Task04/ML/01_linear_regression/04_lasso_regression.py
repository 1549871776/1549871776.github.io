import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_new = np.array(1.5).reshape(1, -1)
lasso_reg = Lasso(alpha=0.15)
lasso_reg.fit(X, y)
y_predict = lasso_reg.predict(X_new)
print(y_predict)
print("w0=", lasso_reg.intercept_)
print("w1=", lasso_reg.coef_)

max_iter = 1000000
n_iter_no_change = 10000
sgd_reg = SGDRegressor(penalty='l1', max_iter=max_iter, n_iter_no_change=n_iter_no_change)
sgd_reg.fit(X, y.ravel())
y_predict_1 = sgd_reg.predict(X_new)
print(y_predict_1)
print("w0=", sgd_reg.intercept_)
print("w1=", sgd_reg.coef_)
