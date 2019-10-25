from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

def get_mnist():
	X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

	random_state = check_random_state(0)
	permutation = random_state.permutation(X.shape[0])
	X = X[permutation]
	y = y[permutation]

	# preprocess data
	X /= 255
	X -= X.mean(axis=1, keepdims=True)

	X_train, X_test, y_train, y_test = train_test_split(
	    X, y, train_size=60000, test_size=10000)

	return X_train, X_test, y_train, y_test
		