import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
# each column corresponds to one of input nodes and we have 4 training examples
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

    
# output dataset - has only one output node (as it has only one column)  
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice) 
# if you set np.random.seed(0) then every time you will get random numbers
# set up seed with np.random.seed(1)
np.random.seed(0)

# initialize weights randomly with mean 0
# this will generate 3 rows and 1 column based randomly generated matrix with range from 0 to -1. This is because we have 
syn0 = 2*np.random.random((3,1)) - 1
print(syn0)


# the range function is amazing, see how the outcome changes as you move from 10000 to 700000 and you could see how close 
# the model goes to the expected outcome
for iter in range(1000000):

    # forward propagation
    l0 = X 
 #l0 is the first layer in NN
 
    l1 = nonlin(np.dot(l0,syn0))
#np.dot helps with matrix multiplication. Multiply the first layer with randomly generated mean syn0 (weights). This creates a matrix of 4x1 ( 4 x 3) dot (3 x 1) = (4 x 1) 

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    # to help with error weighted derivative. There are better and more precise techniques, but this captures the intuition
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print ("Output After Training:")
print (l1)
