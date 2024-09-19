# Softmax for batch of input
import numpy as np

layer_outputs=[[4.8, 1.21, 2.385],
               [8.9, -1.81, 0.2],
               [1.41,1.051,0.026]]


exp_values=np.exp(layer_outputs)

# print(exp_values)
# we want sum of indivisual row for normalization 
print(np.sum(layer_outputs, axis=1,keepdims=True))
# for getting out put similar as dimensions we use "keepdims=True"
#axis=None -> all values
#axis=0 ->sum of all column (vertical addition)
#axis=1 -> sum of row (horizontal additon)

norm_values = exp_values / np.sum(exp_values,axis=1,keepdims=True)
# our output is getting overflow (too big) thatswhy we should subtract all value with the max_value among them
print(norm_values)