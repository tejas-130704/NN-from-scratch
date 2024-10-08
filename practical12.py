#Adding softmax
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

#np.random.seed(0)
nnfs.init() 

# X=[[1,2,3,2.5],
#    [2.0,5.0,-1.0,2.0],
#    [-1.5,2.7,3.3,-0.8]]



class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1 , keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self,output,y):
        sample_loses=self.forward(output,y)
        data_loss=np.mean(sample_loses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        sample=len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7, 1-1e-7 )
        if len(y_true.shape) == 1: #1d -> value passed [0,1,1]
            correct_confidence = y_pred_clipped[range(sample), y_true] #this is for to get required confidence interval o/p ->[0.7,0.5,0.9] 
        
        elif len(y_true.shape) == 2: #2d -> One Hot Encoded form [[1,0,0],[0,1,0],[0,0,1]]
            correct_confidence = np.sum(y_pred_clipped*y_true,axis=1 ) # this also give similar o/p as pervious one i.e. [0.7,0.5,0.9] 
        
        negative_log_likelihoods = -np.log(correct_confidence)
        return negative_log_likelihoods

# 1e-7 -> 1 * 10^(-7) or 0.0000001


X,y=spiral_data(100,3)

# here 2 input features means X, y
dense1= Layer_Dense(2,3)
activation1 = Activation_ReLU()


# this is the next Layer therefore output of the pervious layer will be the input of following layer..
#this is the output layer, and there are 3 clases therfore our output of this layer should be 3
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()



dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

# input ->  layer1 -> activation1 -> layer2 -> activation2 -> Loss Function

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss= loss_function.calculate(activation2.output,y)

print("Loss:",loss)