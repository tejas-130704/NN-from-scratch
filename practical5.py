#Batches, Layers, and O
import numpy as np

#inputs=[1,2,3,2.5]
# Inputs are the samples with four features


#Batch Size=4
inputs=[[1,2,3,2.5],
        [2.0,5.0,-1.0,2.0],
        [-1.5,2.7,3.3,-0.8]]

weights=[[0.2,0.8,-0.5,1.0],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.87]]

bias=[2,3,0.5]

#Here to get output we have to do matrix multplication of `weights` `inputs`
#Shape are: weights->(3,4)
#inputs->(3,4)
#the weights.colu mns != inputs.row
#Transpose the matrix np.array().T

output=np.dot(inputs,np.array(weights).T)+bias
print(output) 