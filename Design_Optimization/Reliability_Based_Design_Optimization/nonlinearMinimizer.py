import torch
import time
import numpy as np
import torch.nn as nn

# Nonlinear minimization with equality constraints using the NN weights directly as parameters

class Model(nn.Module):
    def __init__(self,X0):
        super().__init__()
        weights = X0
        self.weights = nn.Parameter(weights) # make weights torch parameters
          
class NonlinearMinimizer:
    def __init__(self,objectiveFunction,constraintFunction,X0,\
                 tol = 1e-6, alpha0 = 1e-6, alphaMultiplier = 10, \
                     maxIterations = 100, displayProgress = True):
        nUnknowns = X0.detach().numel()
        self.nUnknowns = nUnknowns
        nConstraints = 0
        if (constraintFunction):
            h = constraintFunction(X0)
            nConstraints = h.detach().numel()
        self.displayProgress = displayProgress
        self.nConstraints = nConstraints
        self.net = Model(X0)
        self.toleranceError = tol
        self.alpha0 = alpha0
        self.alphaMultiplier = alphaMultiplier
        self.alphaMax = 1e12
        self.maxIterations = maxIterations
        self.optimizer = torch.optim.LBFGS(self.net.parameters(), \
                                           line_search_fn = 'strong_wolfe') 
        self.nFunctionalCalls = 0
        self.objectiveFunction= objectiveFunction # objective function
        self.constraintFunction = constraintFunction # constraint function
      
    def solve(self):
        self.xVec = self.net.weights # Forward pass
        # first scale the objective
        # sample randomly around initial point
        xRand = self.xVec.detach() + torch.rand(self.nUnknowns)
        fRand = self.objectiveFunction(xRand) 
        self.objectiveScaling = abs(fRand.detach())
        prevObjective = self.objectiveFunction(self.xVec)/self.objectiveScaling 
        self.nFunctionalCalls = 1
        alpha = self.alpha0*torch.ones(self.nConstraints) # penalty
        mu = torch.zeros(self.nConstraints)  # lagrange multiplier
        xPrev = (self.xVec.detach()).clone()

        def closure():  # closure needed for second order LBFGS 
            self.optimizer.zero_grad()# Zero gradients
            self.xVec = self.net.weights
            self.f = self.objectiveFunction(self.xVec)/self.objectiveScaling 
            self.nFunctionalCalls = self.nFunctionalCalls+1
            if (self.nConstraints > 0):
               self.h = self.constraintFunction(self.xVec)
            loss = self.f 
            for i in range(self.nConstraints):
                loss = loss + alpha[i]*torch.pow(self.h[i],2) + mu[i]*self.h[i]
            loss.backward(retain_graph=True)# Backward
            return loss
    
        for it in range(self.maxIterations):  
            self.optimizer.step(closure) 
            xErr = torch.norm(self.xVec.detach()-xPrev)/(torch.norm(xPrev)+1e-12)
            xPrev = self.xVec.detach().clone()
            objectiveError = abs(prevObjective-self.f.item())
            constraintError = 0
            for i in range(self.nConstraints):
                constraintError = constraintError + abs(self.h[i].item()) # combine objective and constraint error
            
            totalError = xErr + objectiveError + constraintError
             
            if (self.displayProgress):
                print(f'{it+1}/{self.maxIterations}, f: {self.f.item()*self.objectiveScaling: 0.6f},  totalError: {totalError:.6f}')

            if (objectiveError > 1e4):# need to rescale
                f = self.objectiveFunction(self.xVec)
                self.objectiveScaling = abs(f.detach())
                prevObjective = 1      
                continue
            
            if (np.isnan(self.f.detach())):
                print('*******Did not converge*********')
                print(f'X: { self.net.weights.detach()}')
                print(f'mu: {mu}')
                print(f'alpha: {alpha}')
                print('Try: (1) properly scaling the problem, (2) reducing alpha0, (3) reducing alphaMultiplier')
                break

            if (totalError <  self.toleranceError): 
                break # we are done         
          
            for i in range(self.nConstraints):
                mu[i] = mu[i] + 2*alpha[i]*self.h[i].item() # update lagrange multiplier      
                if (abs(self.h[i].item()) > 0.1*self.toleranceError):
                    alpha[i] = min(self.alphaMax,alpha[i]*self.alphaMultiplier)  #  increase constraint penalty                   

            prevObjective = self.f.item()
        return [self.net.weights.detach().numpy(),self.f.item()*self.objectiveScaling ,self.nFunctionalCalls]

## Example objective functions
class Objectives:
    def __init__(self):
       self.dummy = 0 
    def objective1(xVec):
        x = xVec[0]
        y = xVec[1]
        f = (x-2)*(x-2)+ (y-3)*(y-3)
        return f
    def objective2(xVec):
        x = xVec[0]
        y = xVec[1]
        f = 100*torch.pow(y- x*x,2 )+ torch.pow((1-x),2)
        return f
    def objective3(xVec):
        u = xVec[0]
        v = xVec[1]
        L12 = torch.sqrt(u*u+(1+v)*(1+v));
        L13 = torch.sqrt(u*u+(1-v)*(1-v));
        f = 0.5*(100*torch.pow(L12-1,2) + 50*torch.pow(L13-1,2)) - (10*u+8*v);
        return f
    def objective4(xVec):
        M = xVec.detach().numel()-1
        f = 0;
        for i in range(M):
            f = f + 100*torch.pow(xVec[i+1]-xVec[i]*xVec[i],2) + torch.pow(1-xVec[i],2)
        return f
    def objective5(xVec):
        x = xVec[0]
        y = xVec[1]
        f = x*x+ y*y
        return f
    def objective6(xVec):
        x = xVec[0]
        y = xVec[1]
        z = xVec[2]
        f = x*x + y*y + z*z
        return f
    def objective7(xVec):
        u = xVec[0]
        v = xVec[1]
        L12 = torch.sqrt(u*u+(1+v)*(1+v));
        L13 = torch.sqrt(u*u+(1-v)*(1-v));
        f = 0.5*(100*torch.pow(L12-1,2) + 50*torch.pow(L13-1,2)) - (10*u+8*v);
        return f
    def objective8(xVec):
        x = xVec[0]
        y = xVec[1]
        f = (1e3)*x*x - (1e-3)*y*y -2*x*y
        return f
    def objective9(xVec):
        nPoints = int(xVec.detach().numel()/3)
        pts = torch.reshape(xVec,(3,nPoints))
        f = 0
        for i in range(nPoints):
            pt_i = pts[:,i]
            for j in range(i+1,nPoints):
                pt_j = pts[:,j]
                dist = torch.sqrt(torch.pow(pt_i[0] - pt_j[0],2) + \
                       torch.pow(pt_i[1] - pt_j[1],2) + \
                       torch.pow(pt_i[2] - pt_j[2],2) ) +1e-12# avoid divide by zero
                f = f + 1/dist
        return f
    def objective10(xVec):
        x = xVec[0]
        y = xVec[1]
        f = 1e10*(x*x+ y*y)
        return f
    def objective11(xVec):
        x = xVec[0]
        y = xVec[1]
        f = 1e-10*(x*x+ y*y)
        return f
    def objective12(xVec):
        x = xVec[0]
        y = xVec[1]
        f = (x*x+y*y)
        return f
    
## Example constraint functions    
class Constraints:
    def __init__(self):
       self.dummy = 0 
    def constraint5(xVec):
        print(xVec)
        h = torch.zeros(1)
        x = xVec[0]
        y = xVec[1]
        h[0] = x + y -1
        return h
    def constraint6(xVec):
        h = torch.zeros(2)
        x = xVec[0]
        y = xVec[1]
        z = xVec[2]
        h[0] = x + y + z - 1
        h[1] = x + 2*y + 3*z - 4
        return h
    def constraint7(xVec):
        h = torch.zeros(1)
        u = xVec[0]
        v = xVec[1]
        h[0] = torch.pow(u,3) - v
        return h
    def constraint8(xVec):
        h = torch.zeros(1)
        x = xVec[0]
        y = xVec[1]
        h[0] = 1e3*x*x + 1e-3*y*y - 200
        return h
    def constraint9(xVec):
        nPoints = int(xVec.detach().numel()/3)
        pts = torch.reshape(xVec,(3,nPoints)) 
        h = torch.zeros(nPoints)
        for i in range(nPoints):
            h[i] = pts[0,i]*pts[0,i] + pts[1,i]*pts[1,i] + pts[2,i]*pts[2,i] -1;
        return h
    def constraint10(xVec):
        h = torch.zeros(1)
        x = xVec[0]
        y = xVec[1]
        h[0] = x + y -1
        return h
    def constraint11(xVec):
        h = torch.zeros(1)
        x = xVec[0]
        y = xVec[1]
        h[0] = x + y -1
        return h
    def constraint12(xVec):
        h = torch.zeros(1)
        x = xVec[0]
        y = xVec[1]
        h[0] = x + y - 1e6
        return h
# Test nonlinear minimizer    
def test():
    example = 5
    if (example == 1): 
        # Min (x-2)^2 + (y-3)^2
        X0 = torch.zeros(2) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective1,None,X0)
        xExact = [2,3]
        fExact = 0
    elif (example == 2): 
        # 2-dimensional  Rosenbrock
        # Min 100*(y-x^2)^2 + (1-x)^2 
        X0 = torch.zeros(2) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective2,None,X0)
        xExact = [1,1]
        fExact = 0
    elif (example == 3): 
        # unconstrained 2-spring system
        X0 = torch.zeros(2) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective3,None,X0)
        xExact = [0.539505204380469,   0.016644764748537]
        fExact =  -4.019420438041528
    elif (example == 4): 
        #  N-Dimensional Rosenbrock
        X0 = torch.zeros(10) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective4,None,X0)
        xExact = np.ones(10)
        fExact = 0
    elif (example == 5): 
        # Min x^2 + y^2
        # x + y - 1 = 0
        X0 = torch.zeros(2) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective5,Constraints.constraint5,X0)
        xExact = [0.5,   0.5]
        fExact = 0.5
    elif (example == 6): 
        # Min x^2 + y^2 + z^2
        # x + y + z - 1 = 0
        # x + 2*y + 3*z - 4 = 0
        X0 = torch.zeros(3) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective6,Constraints.constraint6,X0)
        xExact = [-2/3,1/3,4/3]
        fExact = 7/3
    elif (example == 7): 
        # Min PE of 2-spring system such that
        # u^3 - v =0
        X0 = torch.zeros(2) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective7,Constraints.constraint7,X0)
        xExact = [ 0.431522334087447,  0.080354431822703]
        fExact = -3.6177
    elif (example == 8): # badly scaled
        # Min 1e4*x*x - 1e-4*y*y -2*x*y
        # 1e4*x*x + 1e-4*y*y - 200= 0
        X0 = torch.ones(2) 
        problem = NonlinearMinimizer(Objectives.objective8,Constraints.constraint8,X0)
        xExact = [0.171141278789652,   4.131715847437702e+02]
        fExact = -282.84284
    elif (example == 9): # Thompson charge problem 
        # Distribute N particles on a unit sphere as far from each other as possible 
        nPoints = 8
        X0 = torch.rand(3*nPoints) 
        #With the default parameters, the solution does not converge
        problem = NonlinearMinimizer(Objectives.objective9,Constraints.constraint9,X0,\
                                     alphaMultiplier = 2)
        xExact = []
        fExact = 19.6752
    elif (example == 10): 
        # Min 1e10(x^2 + y^2)
        # x + y - 1 = 0
        X0 = torch.zeros(2) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective10,Constraints.constraint10,X0)
        xExact = [0.5, 0.5]
        fExact = 5e9
    elif (example == 11): 
        # Min 1e-10(x^2 + y^2)
        # x + y - 1 = 0
        X0 = torch.zeros(2) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective11,Constraints.constraint11,X0)
        xExact = [0.5, 0.5]
        fExact = 5e-11
    elif (example == 12): 
        # Min (x^2 + y^2)
        # x + y - 1e6 = 0
        X0 = torch.zeros(2) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective12,Constraints.constraint12,X0)
        xExact = [0.5e6, 0.5e6]
        fExact = 0.5e12
        
    beginTime = time.time()
    [xMin,fMin,nFunctionCalls] = problem.solve()
    endTime = time.time()
    print('--------------------')
    print(f'xExact: {xExact}')
    print(f'fExact: {fExact}')
    print('--------------------')
    print(f'xMin: {xMin}')
    print(f'fMin: {fMin}')
    print(f'nFunctionCalls: {nFunctionCalls}')
    print(f'time: {endTime-beginTime}')
    
#test()