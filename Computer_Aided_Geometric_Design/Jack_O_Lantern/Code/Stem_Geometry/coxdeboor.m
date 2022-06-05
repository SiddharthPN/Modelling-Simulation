function [Q]= coxdeboor(P,t,k,I,u)
%syms p0 p1 p2 p3 p4 p5 p6 u d1 d2 Q;
%P=[p0 p0; p1 p1;p2 p2;p3 p3;p4 p4;p5 p5];  %control point
%t=[ 0 0 0 0 0.4 0.6 0.6 1 1 1 1 ]; % knot vector
%k=4; % order
%I=5; % intervalNo. which u belongs to
%u=0.456;
j=1:k;
a(j,:)=P(I-k+j,:); % the k points which can affect P(u)
for r=1:k-1        % compute r degree of P
    for j=k:-1:r+1  % compute in each r degree the Pl value
        i=I-k+j;
        d1=u-t(i);
        d2=t(i+k-r)-u;
        a(j,:) = (d1*a(j,:) + d2*a(j-1,:))/(d1+d2);  % cox-de boor method
    end
end

Q=a(k,:); % get the p(u)