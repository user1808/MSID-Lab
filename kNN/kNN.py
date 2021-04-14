import warnings
import os
import gzip
import numpy as np


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def hamming_distance(X, X_train):
    n1 = np.shape(X)[0]
    n2 = np.shape(X_train)[0]
    d = np.shape(X)[1]
    to_return = np.zeros(shape=(n1, n2))
    for i in range(n1):
        print(i, n1)
        for j in range(n2):
            to_return[i][j] = np.sum(np.abs(X[i] - X_train[j]))

    return to_return
    pass


def sort_train_labels_knn(Dist, y):
    w = Dist.argsort(kind='mergesort')
    return y[w]
    pass


def p_y_x_knn(y, k):
    points_number = 10
    result_matrix = []
    for i in range(np.shape(y)[0]):
        helper = []
        for j in range(k):
            helper.append(y[i][j])
        line = np.bincount(helper, None, points_number)
        result_matrix.append([line[0] / k, line[1] / k, line[2] / k, line[3] / k,
                              line[4] / k, line[5] / k, line[6] / k, line[7] / k,
                              line[8] / k, line[9] / k])
    return result_matrix
    pass


def classification_error(p_y_x, y_true):
    n = len(p_y_x)
    m = len(p_y_x[0])
    res = 0
    for i in range(0, n):
        if (m - np.argmax(p_y_x[i][::-1]) - 1) != y_true[i]:
            res += 1
    return res / n
    pass


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    errors = []
    dist = hamming_distance(X_val, X_train)
    sort = sort_train_labels_knn(dist, y_train)
    for i in range(0, len(k_values)):
        errors.append(classification_error(p_y_x_knn(sort, k_values[i]), y_val))

    best_error = min(errors)
    best_k = k_values[np.argmin(errors)]
    return best_error, best_k, errors
    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    X_train, y_train = load_mnist('', kind='train')
    X_test, y_test = load_mnist('', kind='t10k')
    X_train = X_train[0:60000, :]
    y_train = y_train[0:60000]
    X_test = X_test[0:10000, :]
    y_test = y_test[0:10000]
    error,_,_ = model_selection_knn(X_test, X_train, y_test, y_train, [1])
    print("Accuracy:", 1 - error)
