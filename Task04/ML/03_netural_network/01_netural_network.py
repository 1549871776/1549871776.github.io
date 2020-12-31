from sklearn.neural_network import MLPClassifier

X = [[0, 0], [1, 1]]  # 输入层2个节点
y = [0, 1]

clf = MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), max_iter=2000, tol=1e-4)  # 采用随机梯度下降求最优解，隐藏层2个分别有为5个节点和2个节点

clf.fit(X, y)

predicted_value = clf.predict([[2, 2], [-1, -2]])
print(predicted_value)

predicted_prob = clf.predict_proba([[2, 2], [-1, -2]])
print(predicted_prob)

print([coef.shape for coef in clf.coefs_])
print([coef for coef in clf.coefs_])
