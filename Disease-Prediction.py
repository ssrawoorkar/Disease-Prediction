import numpy as np
import pandas as pd

data = pd.read_csv('Testing.csv')

print(data.head())

data = data.dropna()


def label_encode(series):
    unique_classes = series.unique()
    class_map = {cls: idx for idx, cls in enumerate(unique_classes)}
    return series.map(class_map), class_map


data['prognosis'], class_map = label_encode(data['prognosis'])

data = data.to_numpy()

X = data[:, :-1]
y = data[:, -1].astype(int)


def train_test_split(X, y, test_size=0.2):
    np.random.seed(42)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_size = int(test_size * X.shape[0])
    train_indices, test_indices = indices[:-test_size], indices[-test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def one_hot_encode(y, num_classes):
    one_hot_y = np.zeros((num_classes, y.size))
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y


num_classes = len(class_map)
y_train_onehot = one_hot_encode(y_train, num_classes)
y_test_onehot = one_hot_encode(y_test, num_classes)

X_train = X_train.T
y_train_onehot = y_train_onehot
X_test = X_test.T
y_test_onehot = y_test_onehot


def init_params(input_size, hidden_layer_size, output_size):
    w1 = np.random.randn(hidden_layer_size, input_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((hidden_layer_size, 1))
    w2 = np.random.randn(output_size, hidden_layer_size) * np.sqrt(2. / hidden_layer_size)
    b2 = np.zeros((output_size, 1))
    return w1, b1, w2, b2


def ReLu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def forward_propagation(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = ReLu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def dv_ReLu(z):
    return z > 0


def back_propagation(z1, a1, z2, a2, w2, x, y):
    c = y.shape[1]
    dz2 = a2 - y
    dw2 = 1 / c * dz2.dot(a1.T)
    db2 = 1 / c * np.sum(dz2, axis=1, keepdims=True)
    dz1 = (w2.T.dot(dz2)) * dv_ReLu(z1)
    dw1 = 1 / c * dz1.dot(x.T)
    db1 = 1 / c * np.sum(dz1, axis=1, keepdims=True)
    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2


def get_pred(a2):
    return np.argmax(a2, axis=0)


def get_acc(prediction, y):
    return np.sum(prediction == np.argmax(y, axis=0)) / y.shape[1]


def gradient_descent(X, Y, iterations, alpha):
    input_size = X.shape[0]
    hidden_layer_size = 20
    output_size = Y.shape[0]
    w1, b1, w2, b2 = init_params(input_size, hidden_layer_size, output_size)
    for i in range(iterations):
        z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = back_propagation(z1, a1, z2, a2, w2, X, Y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 10 == 0:
            print("iteration: ", i)
            print("accuracy: ", get_acc(get_pred(a2), Y))
    return w1, b1, w2, b2


# Train the model
iterations = 2000
learning_rate = 0.01
w1, b1, w2, b2 = gradient_descent(X_train, y_train_onehot, iterations, learning_rate)
