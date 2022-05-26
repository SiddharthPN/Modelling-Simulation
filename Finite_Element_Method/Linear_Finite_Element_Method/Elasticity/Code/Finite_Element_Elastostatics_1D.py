import numpy as np
import matplotlib.pyplot as plt
import math

# Author: Siddharth Palani Natarajan 

# ME 601 - Finite Element Method for Elastostatics in 1D

#########################################################

# Section 1: Input Parameters
# Section 2: Analytical Solution
# Section 3: Functions
# Section 4: Finite Element Method

#########################################################

############# Section 1: Input Parameters ###############

#########################################################

# Geometric Data

E = 1e11          # Young's Modulus
Ae = 1e-4         # Area of Element
L = 0.1           # Length
F_til = 1e6       # Body Force
F_bar = 1e7       # Body Force
g1 = 0            # Dirichlet Boundary Condition 1
g2 = 0.001        # Dirichlet Boundary Condition 2
h = 1e6           # Neumann Boundary Condition (Traction)
dim = 1           # Dimension of Problem

#########################################################

########### Section 2: Analytical Solution ##############

#########################################################

x = np.arange(0,L + L/100,L/100)

# Boundary Condition 1

D1 = (g2/L)*x

# Boundary Condition 2

D2 = (F_til/(2*Ae*E))*(L*x - x*x) + (g2/L)*x

# Boundary Condition 3

D3 = (1/(Ae*E))*((-F_til/2)*(x*x) + (h + (F_til*L))*x)

# Boundary Condition 4

D4 = (F_bar/(6*Ae*E))*((L*L)*x - x*x*x) + (g2/L)*x

#########################################################

############### Section 3: Functions ####################

#########################################################

def FEM1D(Nel, order, Nq, BC):

     nen = order + 1;   # Number of nodes per element (nen = 2,3 & 4)
     Tnodes = (Nel-1)*(nen-1) + nen;   # Total number of nodes 
     
     Le = L/Nel

     F = F_til;         # BC = 2 & 3
    
     if BC == 1:        # BC = 1
         F = 0

     ECA = Element_Connectivity_Array(Nel,nen)

     NCA = Nodal_Connectivity_Array(L,Tnodes)

     Quadrature = Gauss_Quadrature(Nq)

     # Initializing Global and Dirichlet Matrices

     K_Global = np.zeros([Tnodes*dim,Tnodes*dim])
     F_Global = np.zeros([Tnodes*dim,1])
     d_Global = np.zeros([Tnodes*dim,1])
    
     K_GlobalD = np.zeros([Tnodes*dim-1,Tnodes*dim-1])
     F_GlobalD = np.zeros([Tnodes*dim-1,1])
     d_GlobalD = np.zeros([Tnodes*dim-1,1])


     # Element Loop

     for e in range(Nel):

         K_Local = np.zeros([nen,1])
         F_Local = np.zeros([nen,1])

         # Quadrature Loop

         for q in range(Nq):

             z = Quadrature[q,0]
             w = Quadrature[q,1]

             N, dNdz = Shape_Functions(order,z)

             Jacobian = Le/2
             Jinv = 1/Jacobian

             K_LocalQ = np.zeros([nen,nen])
             for A in range(nen):
                 for B in range(nen):
                     K_LocalQ[A,B] = dNdz[A]*dNdz[B]*(E*Ae*Jinv)

            
             F_LocalQ = np.zeros([nen,1])
             for A in range(nen):
                 if BC == 4:
                     mid = ((e + 1) - 1)*Le + Le/2
                     xz = z*(Le/2) + mid
                     F = 1e7*xz
                 F_LocalQ[A,0] = N[A]*F*Jacobian
                 
                 
             K_Local = K_Local + K_LocalQ*w
             F_Local = F_Local + F_LocalQ*w

         # Assembly Process

         x = ECA[e][0] - 1
         i = int(x)

         if nen == 2:

            # Stiffness Matrix Assembly

            K_Global[i][i] += K_Local[0][0]
            K_Global[i][i+1] += K_Local[0][1]
            K_Global[i+1][i] += K_Local[1][0]
            K_Global[i+1][i+1] += K_Local[1][1]
            
            # Force Matrix Assembly

            F_Global[i][0] += F_Local[0][0]
            F_Global[i+1][0] += F_Local[1][0]
                
         if nen == 3:

            # Stiffness Matrix Assembly

            K_Global[i][i] += K_Local[0][0]
            K_Global[i][i+1] += K_Local[0][1]
            K_Global[i][i+2] += K_Local[0][2]
            K_Global[i+1][i] += K_Local[1][0]
            K_Global[i+1][i+1] += K_Local[1][1]
            K_Global[i+1][i+2] += K_Local[1][2]
            K_Global[i+2][i] += K_Local[2][0]
            K_Global[i+2][i+1] += K_Local[2][1]
            K_Global[i+2][i+2] += K_Local[2][2]
            
            # Force Matrix Assembly

            F_Global[i][0] += F_Local[0][0]
            F_Global[i+1][0] += F_Local[1][0]
            F_Global[i+2][0] += F_Local[2][0]

     # Applying Dirichlet Boundary Conditions

     if BC == 3:
        F_Global[Tnodes-1][0] += h
        F_GlobalD = np.delete(F_Global,0,0)
        Temp1 = np.delete(K_Global,0,0)
        Temp1 = Temp1[:,0]
        for i in range(Tnodes-1):
            F_GlobalD[i,0] = F_GlobalD[i,0] - Temp1[i]*g1
        
        Temp2 = np.delete(K_Global,0,0)
        Temp2 = np.delete(Temp2,0,1)
        K_GlobalD = Temp2
        
     else:
        F_GlobalD = np.delete(F_Global,Tnodes-1,0)
        F_GlobalD = np.delete(F_GlobalD,0,0)
        Temp1 = np.delete(K_Global,Tnodes-1,0)
        Temp1 = np.delete(Temp1,0,0)
        Temp1 = Temp1[:,0]
        
        Temp2 = np.delete(K_Global,Tnodes-1,0)
        Temp2 = np.delete(Temp2,0,0)
        Temp2 = Temp2[:,Tnodes-1]
        for i in range(Tnodes-2):
            F_GlobalD[i,0] = F_GlobalD[i,0] - Temp1[i]*g1 - Temp2[i]*g2
            
        Temp3 = np.delete(K_Global,Tnodes-1,0)
        Temp3 = np.delete(Temp3,Tnodes-1,1)
        Temp3 = np.delete(Temp3,0,0)
        Temp3 = np.delete(Temp3,0,1)
        K_GlobalD = Temp3

     # Solve

     K_GlobalDinv = np.linalg.inv(K_GlobalD)
     d_GlobalD = np.matmul(K_GlobalDinv,F_GlobalD)
    
     d_Global[0,0] = g1
    
     if BC != 3:
        d_Global[Tnodes-1][0] = g2
    
     if BC != 3:
        for i in range(Tnodes-2):
            d_Global[i+1][0] = d_GlobalD[i][0]
        
     if BC == 3:
        d_Global_Temp = np.append(d_Global[0,0],d_GlobalD)
        for kk in range (Tnodes):
            d_Global[kk][0] = d_Global_Temp[kk]

     return d_Global

def Element_Connectivity_Array(Nel, nen):

    ECA = np.zeros([Nel,nen])
    k = 1

    for i in range(Nel):
        for j in range(nen):

            ECA[i,j] = k
            k = k + 1

        k = k - 1

    return ECA

def Nodal_Connectivity_Array(L, Tnodes):

    NCA = np.zeros([Tnodes,1])

    for i in range(Tnodes):

        NCA[i] = (i+1-1)*L/(Tnodes - 1)

    return NCA

def Shape_Functions(order, z):

    if order == 1:

        N = np.zeros(2)
        dNdz = np.zeros(2)

        # Shape Function 

        N[0] = (1 - z)/2
        N[1] = (1 + z)/2

        # Shape Function derivative in zeta

        dNdz[0] = -1/2
        dNdz[1] = 1/2

    elif order == 2:

        N = np.zeros(3)
        dNdz = np.zeros(3)

        # Shape Function 

        N[0] = -(1 - z)*(z/2)
        N[1] = (1 - z)*(1 + z)
        N[2] = +(1 + z)*(z/2)

        # Shape Function derivative in zeta

        dNdz[0] = -(1 - (2*z))/2
        dNdz[1] = -(2*z)
        dNdz[2] = +(1 + (2*z))/2

    return N, dNdz

def Gauss_Quadrature(Nq):

    if Nq == 1:

        Q = np.zeros([1,2])

        Q[0,0] = 0
        Q[0,1] = 2

    elif Nq == 2:

        Q = np.zeros([2,2])

        Q[0,0] = -1/(3**0.5)
        Q[0,1] = 1
        Q[1,0] = +1/(3**0.5)
        Q[1,1] = 1

    elif Nq == 3:

        Q = np.zeros([3,2])

        Q[0,0] = -((3/5)**0.5)
        Q[0,1] = 5/9
        Q[1,0] = 0
        Q[1,1] = 8/9
        Q[2,0] = +((3/5)**0.5)
        Q[2,1] = 5/9

    return Q

def Plot_Solution(FEM, Actual, plot_points1, plot_points2, Label, Title):

    L = 0.1

    dx = L/(plot_points1 - 1)
    x = np.arange(0, L + dx, dx)
    plt.plot(x, FEM, label = Label)
    dx = L/(plot_points2 - 1)
    x = np.arange(0, L + dx, dx)
    plt.plot(x, Actual, label = " Actual Solution ")
    plt.legend()
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(Title)
    plt.show()

#########################################################

######### Section 4: Finite Element Method ##############

#########################################################

# Boundary Condition 1

D1b = FEM1D(100, 1, 1, 1)
Plot_Solution(D1b, D1, 101, 101, " FEM for " + str(100) + " Elements ", "Boundary Condition 1 Order: 1 Quadrature Points: 1")
D1d = FEM1D(100, 2, 2, 1)
Plot_Solution(D1d, D1, 201, 101, " FEM for " + str(100) + " Elements ", "Boundary Condition 1 Order: 2 Quadrature Points: 2")

# Boundary Condition 2

D2b = FEM1D(100, 1, 1, 2)
Plot_Solution(D2b, D2, 101, 101, " FEM for " + str(100) + " Elements ", "Boundary Condition 2 Order: 1 Quadrature Points: 1")
D2d = FEM1D(100, 2, 2, 2)
Plot_Solution(D2d, D2, 201, 101, " FEM for " + str(100) + " Elements ", "Boundary Condition 2 Order: 2 Quadrature Points: 2")

# Boundary Condition 3

D3b = FEM1D(100, 1, 1, 3)
Plot_Solution(D3b, D3, 101, 101, " FEM for " + str(100) + " Elements ", "Boundary Condition 3 Order: 1 Quadrature Points: 1")
D3d = FEM1D(100, 2, 2, 3)
Plot_Solution(D3d, D3, 201, 101, " FEM for " + str(100) + " Elements ", "Boundary Condition 3 Order: 2 Quadrature Points: 2")

# Boundary Condition 4

D4b = FEM1D(100, 1, 1, 4)
Plot_Solution(D4b, D4, 101, 101, " FEM for " + str(100) + " Elements ", "Boundary Condition 4 Order: 1 Quadrature Points: 1")
D4d = FEM1D(100, 2, 2, 4)  
Plot_Solution(D4d, D4, 201, 101, " FEM for " + str(100) + " Elements ", "Boundary Condition 4 Order: 2 Quadrature Points: 2") 


     
  








