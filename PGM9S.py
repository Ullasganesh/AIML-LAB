import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(x0, x, tau):
    m,n = np.shape(x)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff = x0 - Z[j]
        weights[j,j] = np.exp(diff*diff.T/(-2.0*tau**2))
    return weights

def localWeight(x0, x, y, tau):
    xw = kernel(x0,x,tau)
    beta = (Z.T*(xw*Z)).I*(Z.T*(xw*y.T))
    return beta
     
def localWeightRegression(x, y, tau):
    m,n = np.shape(x)
    pred = np.zeros(m)
    for i in range(m):
        pred[i] = x[i]*localWeight(x[i],x,y,tau)
    return pred
       
data = pd.read_csv('dataset9s.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)
 
mbill = np.mat(bill)
mtip = np.mat(tip)

m= np.shape(mbill)[1]
one = np.mat(np.ones(m))
Z= np.hstack((one.T,mbill.T))

pred = localWeightRegression(Z,mtip,0.5)
SortIndex = Z[:,1].argsort(0)
xsort = Z[SortIndex][:,0]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(bill,tip, color='green')
ax.plot(xsort[:,1],pred[SortIndex], color = 'red', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show()