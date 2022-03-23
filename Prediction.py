import numpy as np

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    one_matrix = np.ones((m, 1))
    X = np.append(one_matrix, X, axis=1)
    z2 = np.dot(X, Theta1.transpose())
    a2 = 1 / (1 + np.exp(-z2))
    one_matrix = np.ones((m, 1))
    a2 = np.append(one_matrix, a2, axis=1)
    z3 = np.dot(a2, Theta2.transpose())
    a3 = 1 / (1 + np.exp(-z3))
    p = (np.argmax(a3, axis=1))
    return p