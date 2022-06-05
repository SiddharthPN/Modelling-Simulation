function [P, U] = BsplineInterpolate(D,k,n)
%2D interpolation,
%Could extend to 3D directly
% this is a global interpolation routine
% input: data points D, order of the curve k and 
%   number of data points n
% output: control points P.
%           knot vector U.
% Note: in this interpolation routine, # data points = # control points


% D=[0 0; 3 4; -1 4; -4 0; -4 -3]; % data points
%D=[4 1; 5 2; 6 2;8 2;9 3];
% k=4;
% n=5;

% input: D, k, n
% output: control point of the resulting B-spline curve
%D=[d1 d1; ... ;dn dn];  %control point
%k: order
%n: data point count

%Number of data points should be no less than the Order of the curve
if(k>n) k = n;
end

%STEP 1: Computing parameters
totalChord = 0;
for i=2:n
    totalChord = totalChord + sqrt((D(i-1,1)-D(i,1))^2+(D(i-1,2)-D(i,2))^2);
end

ub(1) = 0.;
for i=2:n-1
    ub(i) = ub(i-1) + sqrt((D(i-1,1)-D(i,1))^2+(D(i-1,2)-D(i,2))^2)/totalChord ;
end
ub(n) = 1.0 ;

%STEp 2: Computing knot vector
for j=1:n-k,
    U(j+k) = 0.;
    for i=j:j+k-2
        U(j+k) = U(j+k)+ub(i+1);% averaging k-1 parameters
    end
    U(j+k) = U(j+k)/(k-1);
end

for j=1:k,
    U(j) = 0.;
end

for j=n+1:n+k,
    U(j) = 1.;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Construct the A matrix 
%And then solve Ax = b

for i=2:n-1
    %Find span
    span = 0;
    while ub(i)>U(span+1)
        span = span+1;
    end
    span = span - 1;
    
    
    % Define the basis function
    [N] = Basisfuns(ub(i),U,span,k);

    for j=0:k-1
        A(i,span-k+2+j) = N(j+1);
    end
end

A(1,1) = 1.0 ;
A(n,n) = 1.0 ;

P(:,1) = A\D(:,1);
P(:,2) = A\D(:,2);

% output U
U
ub


