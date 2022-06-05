function [Q] = bsplineCurve(P, k, t, n)
%P control points of b-spline
%k order of b-spline
%t knot vector of b-spline
%n for display, namely how many points to be computed on every segment

[m,d] = size(P);   %get number of control points m
L = 1;             %index to computed point

for i=(k):(m)       %b-splinbe parameter domain  is t(k) - t(m+1)
    step = (t(i+1)-t(i))/(n-1);   %parameter increment step
    for u=t(i):step:(t(i+1))      %do calculation for every segment
        Q(L,:) = deBoor(k, t, P, u, i); %P, degree, u, knots, i, 0);
        L = L+1;
    end
end