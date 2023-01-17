import numpy as np

# Create probability distributions from theta
def zeroed(arr):
    return arr.clip(min=0)

def prob_dists(inputArray):
    vector = inputArray
    probabilityVector = np.zeros(vector.shape)
    for x in range(vector.shape[0]):
        new_vector = zeroed(vector[x])
        vectorSum = sum(new_vector)
        probabilityVector[[x]] = new_vector / vectorSum
    return probabilityVector 