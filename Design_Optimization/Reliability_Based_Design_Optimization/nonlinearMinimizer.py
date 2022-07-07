import torch
import time
import numpy as np
from torchmin import minimize


AGlobal = 1e3# global scope variable

class NonlinearMinimizer:
    def __init__(self,objectiveFunction,constraintFunction,X0,\
                 tol = 1e-6, alpha0 = 1e-8, alphaMultiplier = 10, \
                     maxIterations = 100, objectiveScaling = 1, 
                     displayProgress = False):

        self.X0 = X0
        self.nUnknowns = X0.detach().numel()
        self.XScaling = np.ones(self.nUnknowns) 
        
    
        xRand = X0.detach() + torch.rand(self.nUnknowns)  
        xRand.requires_grad = True
        fRand = objectiveFunction(xRand) 
        self.objectiveScaling =  abs(fRand.detach())

        fRand.backward()
        gradf = xRand.grad.numpy()
        minGradf = min(abs(gradf))
        maxGradf = max(abs(gradf))
        if (maxGradf > 1e5*minGradf):# Check if the design variables are poorly scaled
            print('**********************************************')
            print(f'minGradf: {minGradf}')
            print(f'maxGradf: {maxGradf}')
            print('Design variables are poorly scaled')
            print('**********************************************')
            for i in range(self.nUnknowns):
                self.XScaling[i] = 1/np.sqrt(abs(gradf[i])) # an attempt at rescaling

            fRand = objectiveFunction(self.XScaling) 
            self.objectiveScaling =  abs(fRand)
            
        self.nConstraints = 0
        if (constraintFunction):
            h = constraintFunction(X0)
            self.nConstraints = h.detach().numel()
            self.alpha = alpha0*torch.ones(self.nConstraints) # penalty
            for i in range(self.nConstraints):
                self.alpha[i] = 1/(abs(h[i].item())+1)  # helps speed up convergence
        
        if (displayProgress):
            print(f'XScaling: {self.XScaling}')
            print(f'objectiveScaling: {self.objectiveScaling}')
            print(f'alpha: {self.alpha}')
        self.displayProgress = displayProgress
        self.toleranceError = tol
        self.alphaMultiplier = alphaMultiplier
        self.alphaMax = 1e8
        self.maxIterations = maxIterations
        self.nFunctionalCalls = 0
        self.objectiveFunction= objectiveFunction # objective function
        self.constraintFunction = constraintFunction # constraint function
        self.mu = torch.zeros(self.nConstraints)  # lagrange multiplier
    
    def functionToMinimize(self,xVec):
        xScaled = torch.zeros(self.nUnknowns) 
        for i in range(self.nUnknowns):
            xScaled[i] = xVec[i]*self.XScaling[i]
        self.f = self.objectiveFunction(xScaled)/self.objectiveScaling
        LAug = self.f
        if (self.nConstraints > 0):
           self.h = self.constraintFunction(xScaled)
           for i in range(self.nConstraints):
               LAug = LAug + self.alpha[i]*(self.h[i]*self.h[i]) + self.mu[i]*self.h[i]
        
        return (LAug)
    
    def solve(self,methodUsed = 'l-bfgs'):
        # Select from the following methods for unconstrained problems:
            #  ['bfgs', 'l-bfgs', 'cg', 'newton-cg', 'newton-exact', 
            #   'trust-ncg', 'trust-krylov', 'trust-exact', 'dogleg']
            
        X0 = self.X0
        tolerance = self.toleranceError
        xErr = fErr = hErr = 0
        if (self.nConstraints == 0):
            result = minimize(self.functionToMinimize, X0, method=methodUsed,tol=tolerance)
            self.nFunctionalCalls = self.nFunctionalCalls + result.nfev
        else:
            xPrev = X0.detach().clone()
            fPrev = (self.functionToMinimize(X0)).detach().numpy()
            for it in range(self.maxIterations):         
                if (self.displayProgress):
                    print(f'------------------------iter: {it}-----------------------')
                    print(f'X0: {X0}')

                result = minimize(self.functionToMinimize, X0, method=methodUsed,tol=tolerance)
                # if (not result.success):
                #     print(result.message)
                self.nFunctionalCalls = self.nFunctionalCalls + result.nfev
                hErr = 0
                for i in range(self.nConstraints):
                    hErr = hErr + abs(self.h[i].item())
                    self.mu[i] = self.mu[i] + 2*self.alpha[i]*self.h[i].item() # update lagrange multiplier      
                    #if (abs(self.h[i].item()) > 0.1*self.toleranceError):
                    self.alpha[i] = min(self.alphaMax,self.alpha[i]*self.alphaMultiplier)  #  increase constraint penalty                   
                
                X0 = result.x.detach().clone()
                xErr = abs(torch.norm(X0.detach()-xPrev)/(torch.norm(xPrev)+1e-12)).numpy()

                fErr = abs((result.fun.item()-fPrev)/(fPrev+1e-12))   
                if (self.displayProgress):
                    print(f'X: {X0}')
                    print(f'xErr: {xErr}')
                    print(f'fErr: {fErr}')
                    print(f'hErr: {hErr}')
                    print(f'alpha: {self.alpha[i] }')
                    print(f'mu: {self.mu[i] }')
                if (hErr < self.toleranceError) and ((xErr < self.toleranceError) or (fErr <  self.toleranceError)):
                    break
                xPrev = X0.detach().clone()
                fPrev = result.fun.item()
        xMin = result.x.clone()
        for i in range(self.nUnknowns):
            xMin[i] = xMin[i]*self.XScaling[i]
        fMin = result.fun.item()*self.objectiveScaling
        for i in range(self.nConstraints):
            self.mu[i] = self.mu[i]*self.objectiveScaling
            
        mu = (self.mu).numpy()
        
        return [xMin,fMin,result.success,self.nFunctionalCalls,result.message,mu]
    
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
        f = (AGlobal*AGlobal)*x*x - (1/(AGlobal*AGlobal))*y*y -2*x*y
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
        f = AGlobal*(x*x+ y*y)
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
    def objective13(xVec):
        x = xVec[0]
        y = xVec[1]
        f = torch.sin(x+y) + (x-y)*(x-y) - 1.5*x + 2.5*y +1
        return f
    def objective14(xVec):
        nPoints = int(xVec.detach().numel()) # intermediate points
        h0 = 1/(nPoints+1) # spacing of grid
        # only the intermediate locations have to be optimized
        a = 1 # starting height
        b = 3 # end height
        f = (xVec[0] + a)/2 * torch.sqrt(1 + torch.pow((xVec[0]-a)/h0,2))
        for i in range(1,nPoints):
            f = f + (xVec[i] + xVec[i-1])/2 * torch.sqrt(1 + torch.pow((xVec[i]-xVec[i-1])/h0,2))
        f = f + (b + xVec[nPoints-1])/2 * torch.sqrt(1 + torch.pow((b-xVec[nPoints-1])/h0,2))
        f = h0*f
        return f
## Example constraint functions    
class Constraints:
    def __init__(self):
       self.dummy = 0 
    def constraint5(xVec):
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
        h[0] = AGlobal*AGlobal*x*x + (1/(AGlobal*AGlobal))*y*y - 1
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
        h[0] = x + y - AGlobal
        return h
    def constraint14(xVec):
        h = torch.zeros(1)
        L0 = 4 # length of chain
        nPoints = int(xVec.detach().numel()) # intermediate points
        h0 = 1/(nPoints+1) # spacing of grid
        # only the intermediate locations have to be optimized
        a = 1 # starting height
        b = 3 # end height    
        L = torch.sqrt(1 + torch.pow((xVec[0]-a)/h0,2))
        for i in range(1,nPoints):
            L = L + torch.sqrt(1 + torch.pow((xVec[i]-xVec[i-1])/h0,2))
        L = L +  torch.sqrt(1 + torch.pow((b-xVec[nPoints-1])/h0,2))
        L = L*h0
        h[0] = L- L0
        return h
# Test nonlinear minimizer    
def test():
    example = 8
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
    elif (example == 8): # badly scaled variables
        # Min (AGlobal*AGlobal)*x*x - y*y /(AGlobal*AGlobal)-2*x*y
        # (AGlobal*AGlobal)*x*x + y*y /(AGlobal*AGlobal)- 1= 0
        X0 = torch.ones(2) 
        problem = NonlinearMinimizer(Objectives.objective8,Constraints.constraint8,X0) 
        xExact = [np.cos(3*np.pi/8)/AGlobal, np.sin(3*np.pi/8)*AGlobal]       
        fExact = Objectives.objective8(xExact)
    elif (example == 9): # Thompson charge problem 
        # Distribute N particles on a unit sphere as far from each other as possible 
        nPoints = 8
        X0 = torch.rand(3*nPoints) 
        problem = NonlinearMinimizer(Objectives.objective9,Constraints.constraint9,X0) 
        xExact = []
        fExact = 19.6752
    elif (example == 10): 
        # Min 1e10(x^2 + y^2)
        # x + y - 1 = 0
        X0 = torch.zeros(2) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective10,Constraints.constraint10,X0) 
        xExact = [0.5, 0.5]
        fExact = 0.5*AGlobal
    elif (example == 11): 
        # Min 1e-10(x^2 + y^2)
        # x + y - 1 = 0
        X0 = torch.zeros(2) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective11,Constraints.constraint11,X0) 
        xExact = [0.5, 0.5]
        fExact = 5e-11
    elif (example == 12): 
        # Min (x^2 + y^2)
        # x + y - 1e6
        X0 = torch.zeros(2) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective12,Constraints.constraint12,X0) 
        xExact = [0.5*AGlobal, 0.5*AGlobal]
        fExact = 0.5*AGlobal*AGlobal
    elif (example == 13): 
        # Min sin(x+y) + (x-y)*(x-y) - 1.5*x + 2.5*y +1
        X0 = torch.zeros(2) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective13,None,X0) 
        xExact = [-0.54719, -1.54719]
        fExact = -1.9133
    elif (example == 14): 
        # hanging chain problem
        nPoints = 100 # intemediate points
        X0 = torch.ones(nPoints) #initialize design variables
        problem = NonlinearMinimizer(Objectives.objective14,Constraints.constraint14,X0) 
        xExact = []
        fExact = 5.068481694
    beginTime = time.time()
    [xMin,fMin,success,nFunctionCalls,message,mu]= problem.solve()
    endTime = time.time()
    print('--------------------')
    print(f'xExact: {xExact}')
    print(f'fExact: {fExact}')
    print('--------------------')
    print(f'xMin: {xMin.numpy()}')
    print(f'fMin: {fMin}')
    print(f'nFunctionCalls: {nFunctionCalls}')
    print(f'mu: {mu}')
    print(f'message: {message}')
    print(f'time: {endTime-beginTime}')
    
#test()