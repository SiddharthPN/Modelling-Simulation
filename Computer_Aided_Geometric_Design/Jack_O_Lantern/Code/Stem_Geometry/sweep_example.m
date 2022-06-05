% Copyright: Xiaoping Qian @ UW-Madison
close all; clear;
% profile 1
P1 = [25 0 5 1;
     15 0 15 1;
     15 0 40 1;
     25 0 50 1;
     25 0 65 1;
     9 0 100 1]; % curve in x-z plane
knots_u = [0 0 0 0 0.33 0.66 1 1 1 1];
order_u = 4;
Q = bsplineCurve(P1, order_u, knots_u, 50);
%do plot of control polygon
plot3(P1(:,1),P1(:,2), P1(:,3),'r-s');
hold on;
plot3(Q(:,1),Q(:,2), Q(:,3),'b','linewidth',2);
hold on;
% profile 2
P2 = [ 0 0 0 1;
      15 0 0 1;
      15 25 0 1;
      0 25 0 1;];
knots_v = [0 0 0 0 1 1 1 1];
order_v = 4;
% P2 = [ 0 0 0 1;
%       15 25 0 1];
% knots_v = [0  0 1 1];
% order_v = 2;
Q = bsplineCurve(P2, order_v, knots_v, 50);
%do plot of control polygon
plot3(P2(:,1),P2(:,2), P2(:,3),'r-s');
plot3(Q(:,1),Q(:,2), Q(:,3),'b','linewidth',2);
hold on;
axis equal;
print('-dpdf','-painters','sweep0.pdf')
a = P2(1,:)-P1(1,:);
CPs = [];
for j=1:size(P2,1)
   
    tCP = P1(:,1:3)+P2(j,1:3); %-a(1:3);
    tCP(:,4) = P1(:,4)*P2(j,4);
    CPs = [CPs; tCP];
end
%plot3(CPs(:,1),CPs(:,2), CPs(:,3),'b-o');
CPs
k1 = order_u;
k2 = order_v;
ncp_u = size(P1,1);
ncp_v = size(P2,1);
cph=[CPs(:,1:3).*CPs(:,4), CPs(:,4)]; % turn into homogneous coorinates
CPs_d3 = reshape(cph',[4,ncp_u,ncp_v]);
% tCP = reshape(cph',[4,ncp_v,ncp_u]);
% CPs_d3 = permute(tCP,[1,3,2]); % change it to u first by switching 3rd dimension 
% w/ 2nd dimension
nrbs_Spink.form='B-NURBS';
nrbs_Spink.dim = 4;
nrbs_Spink.number = [ncp_u, ncp_v];
nrbs_Spink.coefs = CPs_d3;  % CPs in homogeneous coordinates
nrbs_Spink.knots = {knots_u knots_v};
nrbs_Spink.order = [k1 k2];
NrbsSrf2IGES(nrbs_Spink,'sweep.igs','./')
nrbs.type   = 'Surface';
nrbs.number = [ncp_u, ncp_v];
nrbs.coefs = CPs_d3;  % CPs in homogeneous coordinates
nrbs.knots = {knots_u knots_v};
nrbs.order = [k1 k2];
plotNrbs(nrbs); 
%print('-dpdf','-painters','sweep1.pdf')