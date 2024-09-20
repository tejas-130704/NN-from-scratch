import numpy as np
#[dog, cat, human]

# One Hot Encode
# Dog -> [1,0,0] i.e, 1
# Cat -> [0,1,0] i.e, 2
# Human -> [0,0,1] i.e, 3




softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1,0.5,0.4],
                            [0.02,0.9,0.08]])
#predicted O/P (3 O/P) for three input 

# for first 0.7 -> dog
# for first 0.5 -> cat
# for first 0.9 -> cat


class_targets = [0,1,1] # actual output / (I/p) [dog,cat,cat]

print(softmax_outputs[[0,1,2], class_targets]) # to get the largest correlations (bcz of class_targets) for all (i.e, [0.7,0.5,0.9]) 
