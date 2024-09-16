#Activation Function
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

#np.random.seed(0)
nnfs.init() 

X=[[1,2,3,2.5],
   [2.0,5.0,-1.0,2.0],
   [-1.5,2.7,3.3,-0.8]]



#ReLU Activation Function 
    #y=x if x>0
    #y=0 if x<=0
'''
inputs=[0,2,-1,3.3,-2.7,1.1,2.2,-100]
outputs=[]
for i in inputs:
    outputs.append(max(0,i))
    
print(outputs)'''

X,y=spiral_data(100,3)


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)

#n_inputs are the no. of features we pass, here are 2 , which are (x,y)
layer1=Layer_Dense(2,5) 

activation1=Activation_ReLU()

layer1.forward(X) 
# print(layer1.output) 
activation1.forward(layer1.output)
print(activation1.output)

#Whenever we see that the neurons are getting dead (bcz of zeros) we start initallizing Biases