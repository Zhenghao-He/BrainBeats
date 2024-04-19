import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from preprocess import *

class SVM:
    def __init__(self, C=1.0, kernel='linear', tol=1e-3, max_iter=100, gamma=None, coef0=None):
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.b = None
        self.kernel_matrix = None
        self.y = None

        self.gamma = gamma
        self.coef0 = coef0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        print('X')
        print(X)
        # 计算核矩阵
        self.kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.kernel_matrix[i, j] = self._kernel(X[i], X[j])

        # print('kernel_matrix')
        # print(self.kernel_matrix)

        # 初始化alpha和b
        self.alpha = np.random.uniform(low=0, high=self.C, size=n_samples)
        self.b = np.random.uniform(low=-self.C, high=self.C)

        # 计算y值
        self.y = y

        # 进行优化
        num_iter = 0
        while num_iter < self.max_iter:
            alpha_prev = np.copy(self.alpha)

            for i in range(n_samples):
                j = self._select_random_j(i, n_samples)

                # 计算误差
                E_i = self._predict(X[i]) - y[i]
                E_j = self._predict(X[j]) - y[j]

                # 计算上下界
                L = max(0, self.alpha[j] - self.alpha[i])
                H = min(self.C, self.C + self.alpha[j] - self.alpha[i])

                if L == H:
                    continue

                # 计算eta
                eta = 2 * self.kernel_matrix[i, j] - self.kernel_matrix[i, i] - self.kernel_matrix[j, j]
                if eta >= 0:
                    continue

                # 更新alpha_j
                self.alpha[j] -= np.sum(y[j] * (E_i - E_j)) / np.sum(eta)
                # self.alpha[j] -= (y[j] * (E_i - E_j)) / eta
                self.alpha[j] = np.clip(self.alpha[j], L, H)

                # 更新alpha_i
                self.alpha[i] += y[i] * y[j] * (self.alpha[j] - alpha_prev[j])

                # print('update_alpha')
                # print(self.alpha)
                # print(alpha_prev)
                # 更新b
                b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_prev[i]) * self.kernel_matrix[i, i] - y[j] * (
                            self.alpha[j] - alpha_prev[j]) * self.kernel_matrix[i, j]
                b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_prev[i]) * self.kernel_matrix[i, j] - y[j] * (
                            self.alpha[j] - alpha_prev[j]) * self.kernel_matrix[j, j]
                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2

            num_iter += 1

            # 判断收敛条件
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break

        # 保存支持向量
        mask = self.alpha > 0
        self.alpha = self.alpha[mask]
        self.X_sv = X[mask]
        self.y_sv = y[mask]
        print('num_iter:', num_iter)

    def predict(self, X):
        # 预测类别
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            y_pred[i] = np.sum(self.alpha * self.y_sv * self._kernel(X[i], self.X_sv)) + np.sum(self.b)
        print('y_pred')
        print(y_pred)
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = -1
        return y_pred

    def _kernel(self, x1, x2):
        # 计算核函数
        if self.kernel == 'linear':
            return np.inner(x1, x2)
        elif self.kernel == 'rbf':
            sigma = 1 * 10 / np.sqrt(len(x1))
            # print('sigma', sigma)
            # print('x1', x1)
            # print('x2', x2)
            # print('x1和x2的rbf', np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2)))
            # time.sleep(2)
            return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))
        elif self.kernel == 'sigmoid':
            if self.gamma is None:
                self.gamma = 1 / len(x1)
            if self.coef0 is None:
                self.coef0 = 0.0
            return np.tanh(self.gamma * np.inner(x1, x2) + self.coef0)
        else:
            raise ValueError('Unsupported kernel function')

    def _predict(self, x):
        # 计算模型输出
        return np.dot(self.alpha * self.y, self.kernel_matrix[:, self.y == self.y[0]]) + self.b

    def _select_random_j(self, i, n_samples):
        # 随机选择一个不等于i的j
        j = i
        while j == i:
            j = np.random.randint(0, n_samples)
        return j


# 定义超参数列表
C_list = [0.1, 1, 10]
# C_list = [0.1]
kernel_list = ['rbf', 'sigmoid', 'linear']
# kernel_list = ['linear', 'rbf']

print('x_open', x_open.shape)
print('y_open', y_open.shape)
X_train, X_test, y_train, y_test = train_test_split(x_open, y_open, test_size=0.2, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

X_train = X_train / 1e08
X_dev = X_dev / 1e08
X_test = X_test / 1e08
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
y_dev[y_dev == 0] = -1
# 选择最优超参数
best_accuracy = 0
best_C = None
best_kernel = None
for C in C_list:
    for kernel in kernel_list:
        svm = SVM(C=C, kernel=kernel, tol=1e-3, max_iter=10, gamma=1, coef0=0.2)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_dev)
        # print('y_dev', y_dev)
        # print('y_pred', y_pred)
        accuracy = accuracy_score(y_dev, y_pred)
        print(f'C={C}, kernel={kernel}, accuracy={accuracy}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C
            best_kernel = kernel

# 在整个训练集上重新训练模型
svm = SVM(C=best_C, kernel=best_kernel, tol=1e-3, max_iter=10)
svm.fit(X_train, y_train)

# 预测测试集
y_pred = svm.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

X_train, X_test, y_train, y_test = train_test_split(x_close, y_close, test_size=0.2, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

X_train = X_train / 1e08
X_dev = X_dev / 1e08
X_test = X_test / 1e08
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
y_dev[y_dev == 0] = -1
# 选择最优超参数
best_accuracy = 0
best_C = None
best_kernel = None
for C in C_list:
    for kernel in kernel_list:
        svm = SVM(C=C, kernel=kernel, tol=1e-3, max_iter=10, gamma=4, coef0=2)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_dev)
        # print('y_dev', y_dev)
        # print('y_pred', y_pred)
        accuracy = accuracy_score(y_dev, y_pred)
        print(f'C={C}, kernel={kernel}, accuracy={accuracy}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C
            best_kernel = kernel

# 在整个训练集上重新训练模型
svm = SVM(C=best_C, kernel=best_kernel, tol=1e-3, max_iter=10)
svm.fit(X_train, y_train)

# 预测测试集
y_pred = svm.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
