from scipy.io import loadmat
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


mnist = loadmat(r'G:\ML\尚学堂\05_mnist\test_data_home\mldata\mnist-original.mat')
# print(mnist)

x, y = mnist['data_home'].T, mnist['label'].T
# print(x.shape, y.shape)

some_digit = x[36000]
# print(some_digit)
some_digit_image = some_digit.reshape(28, 28)
# print(some_digit_image)

# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
# plt.axis('off')
# plt.show()

x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)  # 打乱顺序
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# print(y_test_5)

sgd_clf = SGDClassifier(loss='log', random_state=42, max_iter=1000, tol=1e-4)
# sgd_clf.fit(x_train, y_train_5.ravel())
# print(sgd_clf.predict([some_digit]))

# skfolds = StratifiedKFold(n_splits=3, random_state=42)
#
# for train_index, test_index in skfolds.split(x_train, y_train_5):
#     clone_clf = clone(sgd_clf)
#     x_train_folds = x_train[train_index]
#     y_train_folds = y_train_5[train_index]
#     x_test_folds = x_train[test_index]
#     y_test_folds = y_train_5[test_index]
#
#     clone_clf.fit(x_train_folds, y_train_folds.ravel())
#     y_pred = clone_clf.predict(x_test_folds)
#     print(y_pred)
#     n_correct = sum(y_pred == y_test_folds.ravel())
#     print('正确率:', n_correct / len(y_pred))

print(cross_val_score(sgd_clf, x_train, y_train_5.ravel(), cv=3, scoring='accuracy'))
print(cross_val_score(sgd_clf, x_train, y_train_5.ravel(), cv=3, scoring='precision'))
print(cross_val_score(sgd_clf, x_train, y_train_5.ravel(), cv=3, scoring='recall'))


class Never5Classifier(BaseEstimator):
    def fit(self, x, y = None):
        pass

    def predict(self, x):
        return np.zeros((len(x), 1), dtype=bool)


never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, x_train, y_train_5, cv=3, scoring='accuracy'))

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5.ravel(), cv=3)
print(confusion_matrix(y_train_5.ravel(), y_train_pred))

print(precision_score(y_train_5.ravel(), y_train_pred))
print(recall_score(y_train_5.ravel(), y_train_pred))
print(sum(y_train_pred))
print(f1_score(y_train_5.ravel(), y_train_pred))

y_train_5 = y_train_5.ravel()

sgd_clf.fit(x_train, y_train_5)
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

print('***************************************************************************')
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

threshold = 200000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3, method='decision_function')
print(y_scores)

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
print(precisions, recalls, thresholds)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precison')
    plt.plot(thresholds, recalls[:-1], 'r--', label='Recall')
    plt.xlabel("Threshold")
    plt.legend(loc='upper left')
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

y_train_pred_90 = (y_scores > 7000)
print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


plot_roc_curve(fpr, tpr)
plt.show()

print(roc_auc_score(y_train_5, y_scores))

print('***************************************************************************')

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv=3, method='predict_proba')
y_scores_forest = y_probas_forest[:, 1]

fpr_forset, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, 'b', label='SGD')
plt.plot(fpr_forset, tpr_forest, label='Random Forest')
plt.legend(loc='lower right')
plt.show()

print((roc_auc_score(y_train_5, y_scores_forest)))
