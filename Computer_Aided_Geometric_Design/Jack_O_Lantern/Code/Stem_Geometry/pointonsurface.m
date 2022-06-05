function out=pointonsurface(p,q,U,V,CP,t,s)

% POINTONSURFACE computes a point on a nonuniform B-spline surface
% of degree (p,q) at given parameter values (t,s). The surface is
% defined by the control net CP and the sequences of knots U and V.

if (nargin ~= 7)
  error('Incorrect number of arguments');
end

m1 = size(U);
m2 = size(V);

n1 = size(CP,2);
n2 = size(CP,3);

if (m1 ~= (n1 + p + 1))
  error('Inconsistency between number of knots, control points and surface degree');
end

if (m2 ~= (n2 + q + 1))
  error('Inconsistency between number of knots, control points and surface degree');
end

% Find knot intervals where parameters t and s are located
% i = findinterval(n1,p,t,U);
% j = findinterval(n2,q,s,V);

if  t>=U(n1+1),
    span = n1 - 1;
else if t<=U(p+1),
        span = p;
    else
        span = 0;
        while t>U(span+1)
            span = span+1;
        end
        span = span - 1;
    end
end
i = span;

if  s>=V(n2+1),
    span = n2 - 1;
else if s<=V(q+1),
        span = q;
    else
        span = 0;
        while s>V(span+1)
            span = span+1;
        end
        span = span - 1;
    end
end
j = span;

% Evaluate basis functions at t and s
% Nt = basisfunc(i,p,t,U);
% Ns = basisfunc(j,q,s,V);

Nt = Basisfuns(t,U,i,p+1);
Ns = Basisfuns(s,V,j,q+1);

% tind = i - p - 1;
% sind = j - q - 1;
 %for j=0:k-1
% A(i,span-k+2+j) = N(j+1);

tind = i - p - 1;
sind = j - q - 1;

pt1 = zeros(size(CP,1),1);
for l = 0 : q    
    for k = 0 : p 
        pt1 = pt1 +  Ns(l+1) * Nt(k+1) * CP(:,i-p+1+k,j-q+1+l);
    end
end

out = pt1;

return
