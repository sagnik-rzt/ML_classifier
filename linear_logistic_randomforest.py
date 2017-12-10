import numpy as np
from scipy.special import expit
from matplotlib import pyplot as plt
from abc import abstractmethod
from sklearn.ensemble import RandomForestClassifier


class ML_classifier():

    def __init__(self):
        pass

    #y(x) is the hypothesis of the classifier for a test-input x
    @abstractmethod
    def y(self, x1, x2):
        pass

    #Returns the gradient vector of the cost-function with respect to weights and biases
    @abstractmethod
    def cost_gradient(self, x1, x2, y):
        pass

    #Trains the classifier
    @abstractmethod
    def train_classifier(self):
        pass

class linear_regression(ML_classifier):

    def __init__(self):
        super().__init__()
        self.weights = [0, 0, 0]

    def y(self, x):
        return np.dot(x, self.weights)

    def cost_gradient(self, x_data, y_data):
        hypothesis = self.y(x_data)
        delta = hypothesis - y_data
        gradient = np.dot(x_data.T, delta)/len(y_data)

        return gradient

    def train_classifier(self,x_data, y_data, epochs = 100000, learning_rate = 0.001):
        for i in range(epochs):
            self.weights -= learning_rate * self.cost_gradient(x_data, y_data)

        return self.weights


class logistic_regression(ML_classifier):

    def __init__(self):
        super().__init__()
        self.weights = [0, 0, 0]

    def y(self, x):
        return expit(np.dot(x, self.weights))

    def cost_gradient(self, x_data, y_data):
        hypothesis = self.y(x_data)
        delta = hypothesis - y_data
        gradient = np.dot(x_data.T, delta)/len(y_data)

        return gradient

    def train_classifier(self,  x_data, y_data, epochs = 100000, learning_rate = 0.0001):
        for i in range(epochs):
            self.weights -= learning_rate * self.cost_gradient(x_data, y_data)

        return self.weights


class random_forest(ML_classifier):

    def __init__(self):
        self.classifier = RandomForestClassifier()

    def y(self, x):
        return self.classifier.predict(x)

    def train_classifier(self, x_data, y_data):
        self.classifier.fit(x_data, y_data)


def main():

    np.random.seed(34)
    num_observations = 1000

    cov = [[1, 0.75], [0.75, 1]]
    mean1 = [0, 0]
    mean2 = [-3, 1]

    one = np.ones((num_observations, 1))
    a = np.random.multivariate_normal(mean1, cov, num_observations)
    b = np.random.multivariate_normal(mean2, cov, num_observations)
    x1 = np.concatenate((a, one), axis = 1)
    x2 = np.concatenate((b, one), axis = 1)

    x_data = np.vstack((x1,x2))
    y_data = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
    x_test = np.random.randn(2000, 3)

    classifier = random_forest()
    classifier.train_classifier(x_data, y_data)
    y_predicted = classifier.y(x_test)

    #plt.figure(figsize = (11, 7))
    #plt.scatter(features[:, 0], features[:, 1], c = y_data, alpha = 0.33)
    #x = np.arange(-5, 5, 0.5)
    #plt.plot(x, -1 * (weights[2] + x * weights[0])/ weights[1] , 'g--')
    #plt.show()


main()