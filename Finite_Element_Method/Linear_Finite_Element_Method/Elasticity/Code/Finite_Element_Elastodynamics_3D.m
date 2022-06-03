% Author: Siddharth Palani Natarajan

% ME 601 - Finite Element Method for Elastodynamics with Rayleigh Damping in 3D

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Section 1 : Input Parameters
% Section 2 : Finite Element Method
% Section 3 : Functions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% Section 1: Input Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Geometric Data

E = 1000;                   % Young's Modulus
nu = 0.3;                   % Poisson's Ratio
Rho = 1;                    % Density
lambda = E/2/(1+nu);        % Lame Parameter
mu = E*nu/(1+nu)/(1-2*nu);  % Lame Parameter
delta = eye(3);             % Kronecker Delta
a = 1;                      % Rayleigh Damping Constant
b = 0.001;                  % Rayleigh Damping Constant
Domain = [3,3,10];          % Domain of Geometry
dim = 3;                    % Dimension of Problem

% Transient State Data

Transient = 1;         % 0: Static, 1: Transient
Time_steps = 100;      % Number of Timesteps
tDelta = 1000;         % Delta T
beta = 0.255;          % Time stepping parameter
gamma = 2*beta;        % Time stepping parameter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%  Section 2: Finite Element Method %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[NCA,ECA,Dirichlet,Elements,Quadrature,nen,u0,v0] = Mesh_Generation(Domain);

[Jacobian,K_Global,M_Global,K_GlobalD,M_GlobalD,F_GlobalD,d_Global] = Assembly(NCA,ECA,Dirichlet,Elements,nen,Rho,lambda,mu,delta,tDelta,a,b,beta,gamma,Time_steps,Quadrature,u0,v0,Transient);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%  Section 3: Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [NCA,ECA,Dirichlet,Elements,Quadrature,nen,u0,v0] = Mesh_Generation(Domain)

% Nodal Connectivity Array

Quadrature = 2;
nen = 8;

Nelx = 4; Nely = 4; Nelz = 40;

Elements = Nelx*Nely*Nelz;

X = Domain(3)/Nelz;
Y = Domain(2)/Nely;
Z = Domain(1)/Nelx;
    
for i=1:Nelz+1
    for j=1:Nely+1
        for k=1:Nelx+1
            NCA(k+(j-1)*(Nelx+1)+(i-1)*(Nelx+1)*(Nely+1),1) = (k-1)*Z; % X - Coordinate
            NCA(k+(j-1)*(Nelx+1)+(i-1)*(Nelx+1)*(Nely+1),2) = (j-1)*Y; % Y - Coordinate
            NCA(k+(j-1)*(Nelx+1)+(i-1)*(Nelx+1)*(Nely+1),3) = (i-1)*X; % Z - Coordinate
        end
    end
end

% Element Connectivity Array
    
for i = 1:Nelz
    for j = 1:Nely
        for k = 1:Nelx
            l = k+(j-1)*Nelx+(i-1)*Nelx*Nely;
                
            ECA(l,1) = l+(j-1)+(i-1)*(Nelx+Nely+1);
            ECA(l,2) = l+(j)+(i-1)*(Nelx+Nely+1);
            ECA(l,3) = l+(j)+Nelx+1+(i-1)*(Nelx+Nely+1);
            ECA(l,4) = l+(j)+Nelx+(i-1)*(Nelx+Nely+1); 
            ECA(l,5) = l+(j-1)+(i)*(Nelx+Nely+1)+Nelx*Nely;
            ECA(l,6) = l+(j)+(i)*(Nelx+Nely+1)+Nelx*Nely;
            ECA(l,7) = l+(j)+Nelx+1+(i)*(Nelx+Nely+1)+Nelx*Nely;
            ECA(l,8) = l+(j)+Nelx+(i)*(Nelx+Nely+1)+Nelx*Nely;   
        end
    end
end

% Initial Condition

Dof = 3;

Tnodes = length(NCA);

u0 = zeros(Dof*Tnodes,1); 
v0 = zeros(Dof*Tnodes,1);

% Dirichlet Array

j = 1;
    
for i = 1:Tnodes
    if(NCA(i,3) == 0)
        Dirichlet(j,1:2) = [3*(i-1)+1,0];
        Dirichlet(j+1,1:2) = [3*(i-1)+2,0];
        Dirichlet(j+2,1:2) = [3*(i-1)+3,0];
        j = j + 3;
    end

    if(NCA(i,3) == Domain(3))
        Dirichlet(j,1:2) = [3*(i-1)+2,0.05];
        j = j + 1;
    end

end

end

function [Jacobian,K_Global,M_Global,K_GlobalD,M_GlobalD,F_GlobalD,d_Global] = Assembly(NCA,ECA,Dirichlet,Elements,nen,Rho,lambda,mu,delta,tDelta,a,b,beta,gamma,Time_steps,Quadrature,u0,v0,Transient)

GaussianMatrix = Gauss_Quadrature(Quadrature);

Tnodes = length(NCA);
Dof = 3;

C = zeros(3,3,3,3);

for i = 1:3
    for j = 1:3
        for k = 1:3
            for l = 1:3
                C(i,j,k,l) = lambda*delta(i,j)*delta(k,l) + 2*mu*(delta(i,k)*delta(j,l) + delta(i,l)*delta(j,k));
            end
        end
    end
end

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
            for qz = 1:Quadrature

                [N,dNdz,dNde,dNdg] = Shape_Functions(GaussianMatrix(qx,1),GaussianMatrix(qy,1),GaussianMatrix(qz,1));

                Jacobian = [dNdz*NCA(ECA(e,:),1)   dNde*NCA(ECA(e,:),1)  dNdg*NCA(ECA(e,:),1);   
                            dNdz*NCA(ECA(e,:),2)   dNde*NCA(ECA(e,:),2)  dNdg*NCA(ECA(e,:),2);
                            dNdz*NCA(ECA(e,:),3)   dNde*NCA(ECA(e,:),3)  dNdg*NCA(ECA(e,:),3)];

                Jinv = inv(Jacobian);

                Basis_Gradient = [dNdz',dNde',dNdg'];
                Basis = N';

                % Stiffness Matrix

                for A = 1:nen
                    for B = 1:nen

                        K_LocalQ = zeros(3,3);

                        for i = 1:3
                            for j = 1:3
                                for k = 1:3
                                    for l = 1:3
                                        for J = 1:3
                                            for L = 1:3

                                                N_A = Basis_Gradient(A, :);
                                                N_B = Basis_Gradient(B, :);
                                                
                                                K_LocalQ(i,k) = K_LocalQ(i,k) + N_A(j)*Jinv(j,J)*C(i,j,k,l)*N_B(l)*Jinv(l,L);

                                            end
                                        end
                                    end
                                end
                            end
                        end

                        K_LocalR(3*(A-1)+1:3*A,3*(B-1)+1:3*B) = K_LocalQ;

                    end
                end

                K_Local = K_Local + K_LocalR*det(Jacobian)*GaussianMatrix(qx,2)*GaussianMatrix(qy,2)*GaussianMatrix(qz,2);

                % Mass Matrix

                for A = 1:nen
                    for B = 1:nen

                        M_LocalQ = zeros(3,3);

                        for i = 1:3
                            for k = 1:3

                                NA = Basis(A, :);
                                NB = Basis(B, :);

                                M_LocalQ(i,k) = M_LocalQ(i,k) + NA*Rho*delta(i,k)*NB;

                            end
                        end

                        M_LocalR(3*(A-1)+1:3*A,3*(B-1)+1:3*B) = M_LocalQ;

                    end
                end

                M_Local = M_Local + M_LocalR*det(Jacobian)*GaussianMatrix(qx,2)*GaussianMatrix(qy,2)*GaussianMatrix(qz,2);

                % Force Vector

                F_Local = F_Local + 0;

            end
        end
    end

    % Assembly Process

    for i = 1:nen

        I = 3*(ECA(e,i)-1)+1:3*ECA(e,i);
        F_Global(I,1) = F_Global(I,1) + F_Local(3*(i-1)+1:3*i,1);

        for j = 1:nen

            J = 3*(ECA(e,j)-1)+1:3*ECA(e,j);
            K_Global(I,J) = K_Global(I,J) + K_Local(3*(i-1)+1:3*i,3*(j-1)+1:3*j);
            M_Global(I,J) = M_Global(I,J) + M_Local(3*(i-1)+1:3*i,3*(j-1)+1:3*j);

        end
    end
end

% Applying Dirichlet Boundary Conditions

j = 1;

for i = 1:Dof*Tnodes
    flag = 0;
    for k = 1:length(Dirichlet)
        
        if(i == Dirichlet(k,1))
            flag = 1;
        end
        
    end
    
    if(flag == 0)
        NonDirichlet(j) = i;
        j = j + 1;
    end

end

% Stiffness and Mass Matrix Dirichlet

for i = 1:length(NonDirichlet)
    for j = 1:length(NonDirichlet)

        K_GlobalD(i,j) = K_Global(NonDirichlet(i),NonDirichlet(j));
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

    d = K_GlobalD\F_GlobalD;

    d_Global = zeros(Dof*Tnodes,1);
    d_Global(Dirichlet(:,1)) = Dirichlet(:,2);
    
    d_Global(NonDirichlet) = d;

    for i = 1:Tnodes

        x(i) = d_Global((i-1)*Dof+1);
        y(i) = d_Global((i-1)*Dof+2);
        z(i) = d_Global((i-1)*Dof+3);

    end

    d_Global = [x',y',z'];

    fem_to_vtk_Vector ('FEM_Elastostatics', NCA, ECA, d_Global);


end

if(Transient == 1)

    % Rayleigh Damping

    C_GlobalD =(a*M_GlobalD + b*K_GlobalD);  
    
    d_Global = zeros(Dof*Tnodes,1);
    v_Global = zeros(Dof*Tnodes,1);

    d_Global(:,1) = u0;
    v_Global(:,1) = v0;
    
    d(:,1) = d_Global(NonDirichlet);
    v(:,1) = v_Global(NonDirichlet);
    vdot(:,1) = M_GlobalD\(F_GlobalD - K_GlobalD*d(:,1) - C_GlobalD*v(:,1) );
    
    for i = 1:Time_steps

        d_tilde = d(:,i) + tDelta*v(:,i) + (1-2*beta)/2*tDelta^2*vdot(:,i);
        v_tilde = v(:,i) + (1-gamma)*tDelta*vdot(:,i);
        vdot(:,i+1) = (M_GlobalD+gamma*tDelta*C_GlobalD+beta*tDelta^2*K_GlobalD)\(F_GlobalD - K_GlobalD*d_tilde - C_GlobalD*v_tilde);
        d(:,i+1) = d_tilde + beta*tDelta^2*vdot(:,i+1);
        v(:,i+1) = v_tilde + gamma*tDelta*vdot(:,i+1);
        d_Global(Dirichlet(:,1),i) = Dirichlet(:,2);
        d_Global(NonDirichlet,i) = d(:,i);

    end


    for i = 1:Time_steps

        d_Time_step = d_Global(:,i);

        for j = 1:Tnodes

            x(j) = d_Time_step((j-1)*Dof+1);
            y(j) = d_Time_step((j-1)*Dof+2);
            z(j) = d_Time_step((j-1)*Dof+3);

        end

        d_Time_step = [x',y',z'];

        filename = strcat('FEM_Elastodynamics_',num2str(i));
        fem_to_vtk_Vector (filename, NCA, ECA, d_Time_step);

    end


end

end

function [N,dNdz,dNde,dNdg] = Shape_Functions(zeta,eta,gamma)

% Shape Functions

N(1) = (1-zeta)*(1-eta)*(1-gamma)/8;
N(2) = (1+zeta)*(1-eta)*(1-gamma)/8;
N(3) = (1+zeta)*(1+eta)*(1-gamma)/8;
N(4) = (1-zeta)*(1+eta)*(1-gamma)/8;
N(5) = (1-zeta)*(1-eta)*(1+gamma)/8;
N(6) = (1+zeta)*(1-eta)*(1+gamma)/8;
N(7) = (1+zeta)*(1+eta)*(1+gamma)/8;
N(8) = (1-zeta)*(1+eta)*(1+gamma)/8;

% Shape Function derivative in zeta

dNdz(1) = -(1-eta)*(1-gamma)/8;
dNdz(2) = (1-eta)*(1-gamma)/8;
dNdz(3) = (1+eta)*(1-gamma)/8;
dNdz(4) = -(1+eta)*(1-gamma)/8;
dNdz(5) = -(1-eta)*(1+gamma)/8;
dNdz(6) = (1-eta)*(1+gamma)/8;
dNdz(7) = (1+eta)*(1+gamma)/8;
dNdz(8) = -(1+eta)*(1+gamma)/8;

% Shape Function derivative in eta

dNde(1) = -(1-zeta)*(1-gamma)/8;
dNde(2) = -(1+zeta)*(1-gamma)/8;
dNde(3) = (1+zeta)*(1-gamma)/8;
dNde(4) = (1-zeta)*(1-gamma)/8;
dNde(5) = -(1-zeta)*(1+gamma)/8;
dNde(6) = -(1+zeta)*(1+gamma)/8;
dNde(7) = (1+zeta)*(1+gamma)/8;
dNde(8) = (1-zeta)*(1+gamma)/8;

% Shape Function derivative in gamma

dNdg(1) = -(1-zeta)*(1-eta)/8;
dNdg(2) = -(1+zeta)*(1-eta)/8;
dNdg(3) = -(1+zeta)*(1+eta)/8;
dNdg(4) = -(1-zeta)*(1+eta)/8;
dNdg(5) = (1-zeta)*(1-eta)/8;
dNdg(6) = (1+zeta)*(1-eta)/8;
dNdg(7) = (1+zeta)*(1+eta)/8;
dNdg(8) = (1-zeta)*(1+eta)/8;

end

function [Q] = Gauss_Quadrature(Nq)

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
