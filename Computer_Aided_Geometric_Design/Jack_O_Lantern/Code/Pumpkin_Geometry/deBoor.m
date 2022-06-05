function [R] = deBoor(k, t, P, u, L)
%k order of b-spline
%t knot vector
%P control points with format that every low is a point
%u parameter value of computed point on b-spline curve
%L  index of knot such as t(L) <= u < t(L+1)

for j=1:k
    A(j,:) = P(L-k+j,:);  %the control points that affect the computation of point on curve
end

for r=1:(k-1)   %time to do recursive compuation
    for j=(k):(-1):(r+1)   %do one time computation to get next level control points 
        i = L-k+j;         
        d1 = u - t(i);       %for left term in recursive format
        d2 = t(i+k-r) - u;   %for right term in recursive format
        A(j,:) = (d1*A(j,:) + d2*A(j-1,:))/(d1 + d2);  %carry out computation
    end
end
R = A(k,:);  %return the computed point value

