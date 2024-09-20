# Calculationg loss

import math
softmax_output = [0.7,0.1,0.2] #this is predicted
target_output = [1,0,0]  # target_class = 0

# to calculate loss we have to log(predicted * output)

loss = -(math.log(softmax_output[0])* target_output[0] +
         math.log(softmax_output[1])* target_output[1] +
         math.log(softmax_output[2])* target_output[2])

print(loss)

loss = -math.log(softmax_output[0])
print(loss)

print(-math.log(0.7)) # 0.35 Loss in less comparitively 
print(-math.log(0.5)) # 0.69 Loss increased bcz it it too far from 1
