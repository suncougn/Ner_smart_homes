import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def computeCost(X, y, theta):
    m = y.size
    h = sigmoid(X.dot(theta))
    return (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradientDescent(X, y, theta, alpha, iterations):
    m = y.size
    J_history = np.zeros(iterations)
    
    for i in range(iterations):
        h = sigmoid(X.dot(theta))
        theta = theta - alpha * (1/m) * X.T.dot(h - y)
        J_history[i] = computeCost(X, y, theta)
    
    return theta, J_history

def mapfeatures(data, degree=7):
    x1 = data['Test 1']
    x2 = data['Test 2']
    for i in range(1, degree+1):
        for j in range(i+1):
            data[f'F{i-j}{j}'] = np.power(x1, i-j) * np.power(x2, j)
    data.drop(['Test 1', 'Test 2'], axis=1, inplace=True)
    return data

if __name__ == "__main__":
    data = pd.read_csv('ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])
    data = mapfeatures(data)
    
    matrix = data.values
    m, n = matrix.shape
    
    X = matrix[:, 1:n]
    X = np.c_[np.ones(m), X]
    y = matrix[:, 0:1].flatten()
    
    theta = np.zeros(X.shape[1])
    alpha = 0.1
    iterations = 10000

    J = computeCost(X, y, theta)
    print(f'Với theta khởi tạo = {theta}\nChi phí tính toán ban đầu = {J:.2f}')

    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

    print('Theta tìm được bằng gradient descent:', theta)
    print('Chi phí cuối cùng:', computeCost(X, y, theta))
