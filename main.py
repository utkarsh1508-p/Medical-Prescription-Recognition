from scipy.io import loadmat
import numpy as np
from Model import neural_network
from RandInitialize import initialise
from Prediction import predict
from scipy.optimize import minimize

# Loading mat file
data1 = loadmat('emnist-balanced-train.mat')
data2 = loadmat('emnist-balanced-test.mat')

# Extracting features from mat file
X_train = data1['X']
X_test = data2['X']

# Normalizing the data
X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape((112800, 784), order='F')
X_test = X_test.reshape((18800, 784), order='F')

for i in range(len(X_train)):
    ex = X_train[i, :].reshape((28, 28), order='F')
    ex = ex.reshape((1, 784))
    X_train[i, :] = ex

for i in range(len(X_test)):
    ex = X_test[i, :].reshape((28, 28), order='F')
    ex = ex.reshape((1, 784))
    X_test[i, :] = ex

# Extracting labels from mat file
y_train = data1['y']
y_test = data2['y']

y_train = y_train.flatten()
y_test = y_test.flatten()
print(y_train, y_train.shape)

m = X_train.shape[0]
input_layer_size = 784  # Images are of (28 X 28) px so there will be 784 features
hidden_layer_size = 130
num_labels = 47

# Randomly initialising Thetas
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)

# Unrolling parameters into a single column vector
initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
maxiter = 800
lambda_reg = 0.1  # To avoid overfitting
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)

# Calling minimize function to minimize cost function and to train weights
results = minimize(neural_network, x0=initial_nn_params, args=myargs, options={'disp': True, 'maxiter': maxiter},
                   method="L-BFGS-B", jac=True)

nn_params = results["x"]  # Trained Theta is extracted

# Weights are split back to Theta1, Theta2
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, input_layer_size + 1))  # shape = (100, 785)
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                    (num_labels, hidden_layer_size + 1))  # shape = (10, 101)

# Checking test set accuracy of our model
pred = predict(Theta1, Theta2, X_test)
print('Test Set Accuracy: {:f}'.format((np.mean(pred == y_test) * 100)))

# Checking train set accuracy of our model
pred = predict(Theta1, Theta2, X_train)
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y_train) * 100)))

# Evaluating precision of our model
true_positive = 0
for i in range(len(pred)):
    if pred[i] == y_train[i]:
        true_positive += 1
false_positive = len(y_train) - true_positive
print('Precision =', true_positive / (true_positive + false_positive))

# Saving Thetas in .txt file
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')
