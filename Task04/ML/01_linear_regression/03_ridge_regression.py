import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

ridge_reg = Ridge(alpha=1, solver="auto")
ridge_reg.fit(X, y)
X_new = np.array(1.5).reshape(1, -1)
y_predict = ridge_reg.predict(X_new)
print(y_predict)
print("w0=", ridge_reg.intercept_)
print("w1=", ridge_reg.coef_)

max_iter = 1000000
sgd_reg = SGDRegressor(penalty='l2', max_iter=max_iter)
sgd_reg.fit(X, y.ravel())
y_predict_1 = sgd_reg.predict(X_new)
print(y_predict_1)
print("w0=", sgd_reg.intercept_)
print("w1=", sgd_reg.coef_)