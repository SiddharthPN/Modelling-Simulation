import torch
import time
import numpy as np
from nonlinearMinimizer import NonlinearMinimizer
betaTarget = 3
XVec0 = torch.ones(2, requires_grad=False) 
U0 =  0.0001*torch.ones(3)   
meanx = [1000, 500, 40000]; # mean values of Y load, Z load, and yield strength
stdx = np.sqrt([100, 100, 2000]); # variance transformed to the std deviation

def outerObjective(xVec):
    f = xVec[0]*xVec[1]
    return f
def outerConstraint(xVec):
    global XVec0
    global U0
    XVec0 = xVec.detach().clone() # needed in inner objective 
    NL =  NonlinearMinimizer(innerObjective,innerConstraint,U0,\
                             displayProgress = False)
    [uMin,ufMin,success,nFunctionCalls,message,mu] = NL.solve() 
    U0 =  uMin.detach().clone() # needed for next iteration
    h = torch.zeros(1)
    # U space transformation
    U1 = uMin[0]*stdx[0] + meanx[0]
    U2 = uMin[1]*stdx[1] + meanx[1]
    U3 = uMin[2]*stdx[2] + meanx[2]
    ufMin = (600*U1/(xVec[0]*xVec[1]*xVec[1]) + 600*U2/(xVec[0]*xVec[0]*xVec[1]) - U3);   
    h[0] = (ufMin/betaTarget-1)
    return h
def innerObjective(uVec):
    global XVec0 # This is assigned in outerConstraint
    # U space transformation
    U1 = uVec[0]*stdx[0] + meanx[0]
    U2 = uVec[1]*stdx[1] + meanx[1]
    U3 = uVec[2]*stdx[2] + meanx[2]
    f = (600*U1/(XVec0[0]*XVec0[1]*XVec0[1]) + 600*U2/(XVec0[0]*XVec0[0]*XVec0[1]) - U3);
    return f
def innerConstraint(uVec):
    h = torch.zeros(1)
    h[0] = torch.sqrt(uVec[0]*uVec[0] + uVec[1]*uVec[1] + uVec[2]*uVec[2])-betaTarget
    return h

X0 = 2.5*torch.ones(2, requires_grad=True)    #initialize design variables
NL = NonlinearMinimizer(outerObjective,outerConstraint,X0, \
                        displayProgress = False)
beginTime = time.time()
[xMin,fMin,success,nFunctionCalls,message,mu] = NL.solve()
endTime = time.time()
print('-----------------------------------------------')
print(f'xMin: {xMin}')
print(f'fMin: {fMin}')
print(f'nFunctionCalls: {nFunctionCalls}')
print(f'time: {endTime-beginTime}')