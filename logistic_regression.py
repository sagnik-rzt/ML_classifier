import numpy as np
from scipy.special import expit
from matplotlib import pyplot as plt
from abc import abstractmethod

class ML_classifier():

    def __init__(self):
        pass

    @abstractmethod
    def y(self, x1, x2):
        pass

    @abstractmethod
    def cost_gradient(self, x1, x2, y):
        pass

    @abstractmethod
    def train_gradient_descent(self,x, y, epochs, learning_rate):
        pass


class logistic_regression(ML_classifier):

    def __init__(self):
        super().__init__()
        self.weights = [0, 0, 0]

    def y(self, x_data):
        return expit(np.dot(x_data, self.weights))

    def cost_gradient(self, x_data, y_data):
        hypothesis = self.y(x_data)
        delta = hypothesis - y_data
        gradient = np.dot(x_data.T, delta)/len(y_data)

        return gradient

    def train_gradient_descent(self,  x_data, y_data, epochs = 100000, learning_rate = 0.0001):
        for i in range(epochs):
            self.weights -= learning_rate * self.cost_gradient(x_data, y_data)

        return self.weights



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
    features = np.vstack((a,b))
    x_data = np.vstack((x1,x2))
    y_data = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

    classifier = logistic_regression()
    weights = classifier.train_gradient_descent(x_data, y_data)
    print(weights)

    '''
    plt.figure(figsize = (11, 7))
    plt.scatter(features[:, 0], features[:, 1], c = y_data, alpha = 0.33)
    x = np.arange(-5, 5, 0.5)
    plt.plot(x, -1 * (weights[2] + x * weights[0])/ weights[1] , 'g--')
    '''

main()