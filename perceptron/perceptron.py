import numpy as np


class Perceptron:
    def __init__(self, num_inputs, bias=0, random_seed=None) -> None:
        if(random_seed):
            np.random.seed(random_seed)
        self.__weights = np.random.uniform(-1, 1, num_inputs)
        self.__bias = bias
    @staticmethod
    def to_npndarray(array):
        if(not isinstance(array, np.ndarray)):
            return np.array(array)
        return array

    @staticmethod
    def sigmoid_fn(x) -> float:
        return 1/(1+np.exp(-x))
    
    @staticmethod
    def d_sigmoid_fn(x) -> float:
        sigmoid = 1/(1+np.exp(-x))
        return sigmoid*(1-sigmoid)

    def predict(self, inputs, activation_pass=True) -> float: 
        inputs = self.to_npndarray(inputs)
        weighted_sum = np.sum(np.dot(inputs, self.__weights))+self.__bias
        return self.sigmoid_fn(weighted_sum) if activation_pass else weighted_sum
    
    def fit(self, inputs, targets, iterations, learning_rate=0.001):
        if(len(inputs) == 0 and len(inputs[0]) != len(self.__weights)):
            raise ValueError(f"Inputs don't match number of inputs defined: {len(inputs[0])}, {len(self.__weights)}")

        inputs = self.to_npndarray(inputs)
        targets = self.to_npndarray(targets)

        for epoch in range(iterations):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i], activation_pass=False)
                error = (prediction-targets[i])
                for w1 in range(len(self.__weights)):
                    self.__weights[w1] -= learning_rate*(2*error)*inputs[i][w1]
 
            print(f'Epoch {epoch+1}: ', end='')
            predictions = np.array([self.predict(inpt, activation_pass=False) for inpt in inputs])
            error = np.sum(np.power(targets-predictions,2))
            print(f'error: {error}, weights: {self.__weights}')
                
                