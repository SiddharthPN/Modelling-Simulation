% Author: Siddharth Palani Natarajan

% ME 601 - Finite Element Method for Heat Conduction in 2D

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Section 1 : Input Parameters
% Section 2 : Finite Element Method
% Section 3 : Functions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% Section 1: Input Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Geometric Data

K_bar = 385;           % Conductivity
Kappa = K_bar*eye(2);  % Conductivity tensor
RhoC = 3.8151*1e6;     % Mass
Rho = 6000;            % Density
Domain = [1,1];        % Domain of Geometry
F = 0;                 % Heat Source / Sink
h = 0;                 % Neumann Boundary Condition (Traction)
g1 = 300;              % Dirichlet Boundary Condition 1
g2 = 310;              % Dirichlet Boundary Condition 2
dim = 2;               % Dimension of Problem

% Transient State Data

Transient = 0;         % 0: Static, 1: Transient
Time_steps = 1000;     % Number of Timesteps
Alpha = 0;             % 0: Forward Euler, 0.5: Crank-Nicholson, 1: Backward Euler

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%  Section 2: Finite Element Method %%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[NCA,ECA,Dirichlet,T0,Elements,Quadrature,nen] = Mesh_Generation(Domain);

[K_GlobalD,M_GlobalD,F_GlobalD,T_Global,Jacobian] = Assembly(NCA,ECA,Dirichlet,T0,Elements,nen,Kappa,RhoC,F,Alpha,Time_steps,Quadrature,Transient);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%  Section 3: Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [NCA,ECA,Dirichlet,T0,Elements,Quadrature,nen] = Mesh_Generation(Domain)

Quadrature = 2;
nen = 4;

Nelx = 2; Nely = 2;

Elements = Nelx*Nely;

ElemL = Domain(1)/Nelx;
ElemH = Domain(2)/Nely;

% Nodal Connectivity Array

for i = 1:(Nely+1)
    for j = 1:(Nelx+1)
        NCA(j + (i-1)*(Nelx+1),1) = (j-1)*ElemL;
        NCA(j + (i-1)*(Nelx+1),2) = (i-1)*ElemH;
    end
end

% Element Connectivity Array

for i = 1:Nely
    for j = 1:Nelx
        k = (j + (i-1)*Nelx);
        ECA(k,1) = k + (i-1);
        ECA(k,2) = k + i;
        ECA(k,3) = k + i + Nelx + 1;
        ECA(k,4) = k + i + Nelx;
    end
end

% Initial Condition

Tnodes = length(NCA);

for i = 1:Tnodes

    if(NCA(i,1) < 0.5)
        T0(i) = 300;
    end

    if(NCA(i,1) >= 0.5)
        T0(i) = 300 + 20*(NCA(i,1) - 0.5);
    end

end

% Dirichlet Array

j = 1;

for i = 1:Tnodes

    if(NCA(i,1) == 0)
        Dirichlet(j,1:2) = [i,300];
        j = j + 1;
    end

    if(NCA(i,1) == 1)
        Dirichlet(j,1:2) = [i,310];
        j = j + 1;
    end

end

end

function [K_GlobalD,M_GlobalD,F_GlobalD,T_Global,Jacobian] = Assembly(NCA,ECA,Dirichlet,T0,Elements,nen,Kappa,RhoC,F,Alpha,Time_steps,Quadrature,Transient)

GaussianMatrix = Gauss_Quadrature(Quadrature);

Tnodes = length(NCA);
Dof = 1;

% Initializing Global and Dirichlet Matrices

K_Global = zeros(Dof*Tnodes, Dof*Tnodes);
M_Global = zeros(Dof*Tnodes, Dof*Tnodes);
F_Global = zeros(Dof*Tnodes, 1);

K_GlobalD = zeros(Dof*Tnodes - length(Dirichlet), Dof*Tnodes - length(Dirichlet));
M_GlobalD = zeros(Dof*Tnodes - length(Dirichlet), Dof*Tnodes - length(Dirichlet));
F_GlobalD = zeros(Dof*Tnodes - length(Dirichlet), 1);

% Element Loop

for e = 1:Elements

    K_Local = zeros(Dof*nen, Dof*nen);
    M_Local = zeros(Dof*nen, Dof*nen);
    F_Local = zeros(Dof*nen, 1);

    % Quadrature Loop

    for qx = 1:Quadrature
        for qy = 1:Quadrature

            [N,dNdz,dNde] = Shape_Functions(GaussianMatrix(qx,1),GaussianMatrix(qy,1));

            Jacobian =[dNdz*NCA(ECA(e,:),1)   dNde*NCA(ECA(e,:),1);
                       dNdz*NCA(ECA(e,:),2)   dNde*NCA(ECA(e,:),2)];

            Jinv = inv(Jacobian);

            Basis_Gradient = [dNdz',dNde'];
            Basis = N';

            % Stiffness Matrix

            for A = 1:nen
                for B = 1:nen
                    for J = 1:2
                        for K = 1:2
                            for j = 1:2
                                for k = 1:2

                                    N_A = Basis_Gradient(A, :);
                                    N_B = Basis_Gradient(B, :);

                                    K_Local(A,B) = K_Local(A,B) + N_A(j)*Jinv(j,J)*Kappa(J,K)*N_B(k)*Jinv(k,K)*det(Jacobian)*GaussianMatrix(qx,2)*GaussianMatrix(qy,2);

                                end
                            end
                        end
                    end
                end
            end

            % Mass Matrix

            for A = 1:nen
                for B = 1:nen

                    NA = Basis(A, :);
                    NB = Basis(B, :);

                    M_Local(A,B) = M_Local(A,B) + RhoC*NA*NB*det(Jacobian)*GaussianMatrix(qx,2)*GaussianMatrix(qy,2);

                end
            end

            % Force Vector

            for A = 1:nen

                NA = Basis(A, :);

                F_Local(A,1) = F_Local(A,1) + NA*F*det(Jacobian)*GaussianMatrix(qx,2)*GaussianMatrix(qy,2);

            end
        end
    end

    % Assembly Process

    K_Global(ECA(e,:),ECA(e,:)) = K_Global(ECA(e,:),ECA(e,:)) + K_Local;
    M_Global(ECA(e,:),ECA(e,:)) = M_Global(ECA(e,:),ECA(e,:)) + M_Local;
    F_Global(ECA(e,:),1) = F_Global(ECA(e,:),1) + F_Local;

end

% Applying Dirichlet Boundary Conditions

j = 1;

for i = 1:Tnodes
    flag = 0;
    for k = 1:length(Dirichlet)
        if(i == Dirichlet(k,1))
            flag = 1;
        end
    end
    if(flag == 0)
        NonDirichlet(j,1) = i;
        j = j + 1;
    end
end

% Stiffness and Mass Matrix Dirichlet

for i = 1:length(NonDirichlet)
    for j = 1:length(NonDirichlet)
        K_GlobalD(i,j) = K_Global(NonDirichlet(i), NonDirichlet(j));
        M_GlobalD(i,j) = M_Global(NonDirichlet(i),NonDirichlet(j));
    end
end

% Force Vector Dirichlet

F_dash = zeros(length(NonDirichlet),1);

for i = 1:length(NonDirichlet)
    for j = 1:length(Dirichlet)
        F_dash(i) = K_Global(NonDirichlet(i),Dirichlet(j,1))*Dirichlet(j,2) + F_dash(i);
    end
end

F_GlobalD = F_Global(NonDirichlet) - F_dash;

% Transient State

if(Transient == 0)

    T = K_GlobalD\F_GlobalD;

    T_Global = zeros(Dof*Tnodes,1);
    T_Global(Dirichlet(:,1)) = Dirichlet(:,2);
    
    T_Global(NonDirichlet) = T;

    fem_to_vtk ('FEM_Heat_Conduction_Static', NCA, ECA, T_Global);

end

if(Transient == 1)

    tDelta = Time_Step(Alpha,M_GlobalD,K_GlobalD);

    T_Global = zeros(Dof*Tnodes, Time_steps);
    T_Global(:,1) = T0';
    
    T(:,1) = T_Global(NonDirichlet);
    Tdot(:,1) = M_GlobalD\(F_GlobalD-K_GlobalD*T(:,1));
    
    for i = 1:Time_steps
        T_tilde = T(:,i) + (1-Alpha)*tDelta*Tdot(:,i);
        Tdot(:,i+1) = (M_GlobalD + Alpha*tDelta*K_GlobalD)\(F_GlobalD - K_GlobalD*T_tilde);
        T(:,i+1) = T_tilde + Alpha*tDelta*Tdot(:,i+1);
        T_Global(Dirichlet(:,1),i) = Dirichlet(:,2);
        T_Global(NonDirichlet,i) = T(:,i);
    end
    
    if Alpha == 0
    
        for Time = 1:Time_steps
    
         filename = strcat('FEM_Heat_Conduction_FE_',num2str(Time));
         fem_to_vtk (filename, NCA, ECA, T_Global(:,Time));
    
        end
    
    elseif Alpha == 1
    
        for Time = 1:Time_steps
    
         filename = strcat('FEM_Heat_Conduction_BE_',num2str(Time));
         fem_to_vtk (filename, NCA, ECA, T_Global(:,Time));
    
        end
    
    elseif Alpha == 0.5
    
        for Time = 1:Time_steps
    
         filename = strcat('FEM_Heat_Conduction_CN_',num2str(Time));
         fem_to_vtk (filename, NCA, ECA, T_Global(:,Time));
    
        end
    
    end

end

end

function [tDelta] = Time_Step(Alpha,M_GlobalD,K_GlobalD)

if(Alpha < 0.5)

    E = eig(M_GlobalD\K_GlobalD);
    Emax = max(E);
    tDelta = 2/(1-2*Alpha)/Emax;

end

if(Alpha >= 0.5)

    tDelta = 10;

end

end

function [N,dNdz,dNde] = Shape_Functions(zeta,eta)

% Shape Functions

N(1) = (1-zeta)*(1-eta)/4;
N(2) = (1+zeta)*(1-eta)/4;
N(3) = (1+zeta)*(1+eta)/4;
N(4) = (1-zeta)*(1+eta)/4;

% Shape Function derivative in zeta

dNdz(1) = -(1-eta)/4;
dNdz(2) = (1-eta)/4;
dNdz(3) = (1+eta)/4;
dNdz(4) = -(1+eta)/4;

% Shape Function derivative in eta

dNde(1) = -(1-zeta)/4;
dNde(2) = -(1+zeta)/4;
dNde(3) = (1+zeta)/4;
dNde(4) = (1-zeta)/4;

end

function[Q] = Gauss_Quadrature(Nq)

if(Nq == 1)
    Q(1,1:2) = [0 2];
end

if(Nq == 2)
    Q(1,1:2) = [-1/sqrt(3) 1];
    Q(2,1:2) = [1/sqrt(3) 1];
end

if(Nq == 3)
    Q(1,1:2) = [-sqrt(3/5) 5/9];
    Q(2,1:2) = [0 8/9];
    Q(3,1:2) = [sqrt(3/5) 5/9];
end

end




