#Softmax = (exponential_value / sum of exponential_value)
import math

E= math.e

layer_outputs=[4.8, 1.21,2.385] #output_layer output

# exponential function (y=e**x)
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)

print(exp_values)

#Normalization (taking probability)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print(norm_values)
# the sum of all should be equal to 1 (as in probability)
print(sum(norm_values))