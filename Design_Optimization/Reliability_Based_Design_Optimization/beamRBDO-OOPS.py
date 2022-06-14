import torch
import time
import numpy as np
from nonlinearMinimizer import NonlinearMinimizer

# Class for Reliability Based Design Optimization (RBDO) for a Beam Geometry
class beamRBDO:

    # Constructor Class with Design Variable (xVec) and Reliability Index (betaTarget)
    def __init__(self,betaTarget):

        self.myBetaTarget = betaTarget

    # Outer Objective Function
    def outerObjective(self,xVec):

        f = xVec[0]*xVec[1]

        return f

    # Outer Constraint Function
    def outerConstraint(self,xVec):

        beta_uVec, beta_beta = self.innerOptimization(xVec)

        u = beta_uVec
        beta_t = self.myBetaTarget

        time.sleep(2)

        cineq = torch.zeros(1)
        cineq[0] = (beta_beta - beta_t)

        return cineq

    # Inner Optimization Loop
    def innerOptimization(self,xVec):

        global XVec

        XVec = xVec.detach().clone()

        u0 = 0.0001*torch.ones(3)

        betaNL = NonlinearMinimizer(self.innerObjective,self.innerConstraint,u0)

        [beta_uVec,beta_beta,nFunctionCalls] = betaNL.solve()

        print(f'xVec0: {XVec.detach()}')
        print(f'uMin: {beta_uVec}')
        print(f'ufMin: {beta_beta}')
        print(f'nFunctionCalls: {nFunctionCalls}')

        return beta_uVec, beta_beta

    # Inner Objective Function
    def innerObjective(self,u):

        meanx = [1000, 500, 40000]
        stdx = [100, 100, 2000]

        U1 = u[0]*stdx[0] + meanx[0]
        U2 = u[1]*stdx[1] + meanx[1]
        U3 = u[2]*stdx[2] + meanx[2]

        f = (600*U1/(XVec[0]*XVec[1]*XVec[1]) + 600*U2/(XVec[0]*XVec[0]*XVec[1]) - U3)

        return f

    # Inner Constraint Function
    def innerConstraint(self,u):

        ceq = torch.zeros(1)
        ceq[0] = torch.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) - self.myBetaTarget

        return ceq

    # Optimization Function
    def optimize(self):

        x0 = 2.5*torch.ones(3, requires_grad = True)

        NL = NonlinearMinimizer(self.outerObjective,self.outerConstraint,x0)

        startTime = time.time()
        [xMin,fMin,nFunctionCalls] = NL.solve()
        stopTime = time.time()

        print(f'xMin: {xMin}')
        print(f'fMin: {fMin}')
        print(f'nFunctionCalls: {nFunctionCalls}')
        print(f'Solver Time: {stopTime - startTime}')


# Test Code
beta_target = 3
beam = beamRBDO(beta_target)
beam.optimize()


