from abc import abstractmethod
x1_data = [1,2,3,4,5,6,7,8,9]
x2_data = [100,150,200,250,300,350,400,450,500]
y_data = [-65,-32,27,31,54,73,126,150,238]

alpha = 0.001  # Learning rate

#Initialized zero weights
theta = [0 ,0 ,0 ,0 ,0 ,0 ,0]

#Feature scaling
def scaled_feature(data):
    mean = sum(data)/len(data)
    delta_square = [(element - mean)**2 for element in data]
    sigma = math.sqrt(sum(delta_square)/len(data))
    scaled = [(element - mean)/sigma for element in data]
    
    return scaled


#Select the features used for regression
#We will be using x1, x1^2, x1^3, x2, x2^2, and x2^3 as features
#Thus we have 6 weights and a bias, and thus our theta vector has 7 elements.
x1_data = scaled_feature(x1_data)
x2_data = scaled_feature(x2_data)
x3_data = [element * element for element in x1_data]
x4_data = [element * element for element in x2_data]
x5_data = [element * element * element for element in x1_data]
x6_data = [element * element * element for element in x2_data]

#Parent abstract class
class ML_classifier():

    def __init__(self):
        pass

    @abstractmethod
    def y(self, x1, x2):
        pass

    @abstractmethod
    def cost_gradient(self, x1_points, x2_points, x3_points, x4_points, x5_points, x6_points, y_points):
        pass

    @abstractmethod
    def train_gradient_descent(self, epochs):
        pass

#Subclass
class Linear_regression(ML_classifier):

    def __init__(self):
        super().__init__()

    #Hypothesis of the model
    def y(self, x1, x2):
        return theta[0] + theta[1] * x1 + theta[2] * x2 + theta[3] * x1 * x1 +  theta[4] * x2 * x2 + theta[5] * x1 * x1 * x1 + theta[6] * x2 * x2 * x2

    #Gradient of the cost function
    def cost_gradient(self, x1_points, x2_points, x3_points, x4_points, x5_points, x6_points, y_points):
        total = [0, 0, 0, 0, 0, 0, 0]
    
        for i in range(1, len(x1_points)):
            total[0] += self.y(x1_points[i], x2_points[i]) -  y_points[i]
            total[1] += ( self.y(x1_points[i], x2_points[i]) -  y_points[i] ) * x1_points[i]
            total[2] += ( self.y(x1_points[i], x2_points[i]) -  y_points[i] ) * x2_points[i]
            total[3] += ( self.y(x1_points[i], x2_points[i]) -  y_points[i] ) * x3_points[i]
            total[4] += ( self.y(x1_points[i], x2_points[i]) -  y_points[i] ) * x4_points[i]
            total[5] += ( self.y(x1_points[i], x2_points[i]) -  y_points[i] ) * x5_points[i]
            total[6] += ( self.y(x1_points[i], x2_points[i]) -  y_points[i] ) * x6_points[i]
        
        total[0] = total[0]/len(x1_points)
        total[1] = total[1]/len(x1_points)
        total[2] = total[2]/len(x1_points)
        total[3] = total[3]/len(x1_points)
        total[4] = total[4]/len(x1_points)
        total[5] = total[5]/len(x1_points)
        total[6] = total[6]/len(x1_points)
    
        return total

    #Train using gradient descent
    def train_gradient_descent(self, epochs = 100000):

        temp1 = theta[0]
        temp2 = theta[1]
        temp3 = theta[2]
        temp4 = theta[3]
        temp5 = theta[4]
        temp6 = theta[5]
        temp7 = theta[6]

        for i in range(epochs):
            cost_derivative = self.cost_gradient(x1_data, x2_data, x3_data, x4_data, x5_data, x6_data, y_data)
            temp1 -= alpha * cost_derivative[0]
            temp2 -= alpha * cost_derivative[1]
            temp3 -= alpha * cost_derivative[2]
            temp4 -= alpha * cost_derivative[3]
            temp5 -= alpha * cost_derivative[4]
            temp6 -= alpha * cost_derivative[5]
            temp7 -= alpha * cost_derivative[6]
            theta[0] = temp1
            theta[1] = temp2
            theta[2] = temp3
            theta[3] = temp4
            theta[4] = temp5
            theta[5] = temp6
            theta[6] = temp7

        return theta

classify = Linear_regression()
#Returns the weights learnt via gradient descent
print(classify.train_gradient_descent())