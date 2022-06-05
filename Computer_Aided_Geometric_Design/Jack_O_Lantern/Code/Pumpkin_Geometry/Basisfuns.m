function [N] = Basisfuns(u,U,i,k)
% compute B-spline basis functions
% input: 
%       u (parameter value)
%       p ( degree of the basis function)
%       i (span index)
%       U (knot vectors, satisfying n+k+1, P0, P1, ...Pn)
%output: N (basis function value)
% algorithm source: Piegl A2.2, P 70.
N(1) = 1.0 ;

for j=1:k-1
    left(j+1) = u-U(i+2-j);
    right(j+1) = U(i+j+1)-u;
    saved = 0.0 ;
    for r = 0:j-1
        temp = N(r+1)/(right(r+2)+left(j-r+1));
        N(r+1) = saved+right(r+2)*temp;
        saved = left(j-r+1)*temp;
    end
    N(j+1) = saved ;
end






