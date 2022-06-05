function [P] = InterForSurf(D,k,n,ub,U)
%B-spline curve interpolation,
% this is a global approximation routine
% input: D (data point -- 1D), 
%       n (n=# of data point)
%       k (order of the curve), 
% output: P (control point of the resulting B-spline curve -- 1D)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 1: Construct the A matrix 
%solve Ax = b

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

P = A\D;

