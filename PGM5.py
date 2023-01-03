import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0) 
y = y/100

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

epoch=5000               
lr=0.1                    
inputlayer_neurons = 2    
hiddenlayer_neurons = 3   
output_neurons = 1        

wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
    
    hlay=np.dot(X,wh)
    hlay=sigmoid(hlay+bh)

    outlay=np.dot(hlay,wout)
    outlay=sigmoid(outlay+bout)

    error=(y-outlay)
    gradoutlay=d_sigmoid(outlay)

    erroroutput=error*gradoutlay
    errorhidden=d_sigmoid(hlay)*np.dot(erroroutput,wout.T)

    wout += hlay.T.dot(erroroutput) *lr
    wh += X.T.dot(errorhidden) *lr
        
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,outlay)