# ML-Coding-Questions
### Classical ML:
#### Implement Linear Regression from Scratch? 
- Class `LinearRegression` with `fit` and `predict` methods.
- fit calculates the `slope` and intercept of the regression line using input data.
- predict predicts output values using the trained model.
- In the __main__ block, sample data is used to train the model, and a test data point is used to make a prediction.
``` Python 
class LinearRegression:
    def __init__(self):
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        mean_X = sum(X) / len(X)
        mean_y = sum(y) / len(y)
        numerator = sum((X[i] - mean_X) * (y[i] - mean_y) for i in range(len(X)))
        denominator = sum((X[i] - mean_X) ** 2 for i in range(len(X)))
        self.weight = numerator / denominator
        self.bias = mean_y - self.weight * mean_X

    def predict(self, X):
        return [self.weight * x + self.bias for x in X]

if __name__ == "__main__":
    X = [1, 2, 3, 4, 5]
    y = [2, 3, 4, 5, 6]
    model = LinearRegression()
    model.fit(X, y)
    X_test = [6]
    print("Prediction:", model.predict(X_test))
```

#### Implement Logistic Regression from Scratch. 
- Defines a `LogisticRegression` class for binary classification.
- Utilizes NumPy for numerical computations.
- Constructor initializes `learning rate` and number of `iterations.`
- fit method trains the model using logistic regression.
- predict method predicts class labels based on the trained model.
- Example usage trains the model with sample data and makes predictions on test data.

``` Python 
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = 1 / (1 + np.exp(-linear_model))

            dw = np.dot(X.T, (y_predicted - y)) / len(X)
            db = np.sum(y_predicted - y) / len(X)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = 1 / (1 + np.exp(-linear_model))
        return (y_predicted > 0.5).astype(int)

# Example usage:
if __name__ == "__main__":
    X_train = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
    y_train = np.array([0, 1, 1])

    model = LogisticRegression()
    model.fit(X_train, y_train)

    X_test = np.array([[2, 3, 4]])
    print("Prediction:", model.predict(X_test))
```


