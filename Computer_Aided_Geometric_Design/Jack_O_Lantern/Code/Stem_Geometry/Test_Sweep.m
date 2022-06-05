% Sweep 
close all;
P1x = [0    0.60 6.3 1;
       0.1  0.70 6.3 1/sqrt(2);
       0.1  0.80 6.4 1;
       0.1  0.90 6.5 1/sqrt(2);
       0    1.00 6.5 1;
      -0.1  1.10 6.5 1/sqrt(2);
      -0.1  1.20 6.4 1;
      -0.1  1.30 6.3 1/sqrt(2);
       0    1.40 6.3 1];

knots_u = [0 0 0 0 0 1/2 1/2 1/2 1/2 1 1 1 1 1];
order_u = 5;

Q = bsplineCurve(P1x, order_u, knots_u, 50);
%do plot of control polygon
plot3(P1x(:,1),P1x(:,2), P1x(:,3),'r-s');
plot3(Q(:,1),Q(:,2), Q(:,3),'b','linewidth',2);
hold on;
hold on;


P2x = [0    0.6 6.3 1;
       0.1  0.6 6.3 1/sqrt(2);
       0.1  0.6 6.4 1;
       0.1  0.6 6.5 1/sqrt(2);
       0    0.6 6.5 1;
      -0.1  0.6 6.5 1/sqrt(2);
      -0.1  0.6 6.4 1;
      -0.1  0.6 6.3 1/sqrt(2);
       0    0.6 6.3 1];

knots_v = [0 0 0 0 0 1/2 1/2 1/2 1/2 1 1 1 1 1];
order_v = 5;


Q = bsplineCurve(P2x, order_v, knots_v, 50);
hold on;
plot3(P2x(:,1),P2x(:,2), P2x(:,3),'r-s');
plot3(Q(:,1),Q(:,2), Q(:,3),'b','linewidth',2);
hold on;
axis equal;


P3x = [0    0.45 6.25 1;
       0.15 0.45 6.25 1/sqrt(2);
       0.15 0.45 6.4  1;
       0.15 0.45 6.55 1/sqrt(2);
       0    0.45 6.55 1;
      -0.15 0.45 6.55 1/sqrt(2);
      -0.15 0.45 6.4  1;
      -0.15 0.45 6.25 1/sqrt(2);
       0    0.45 6.25 1];

knots_x = [0 0 0 0 0 1/2 1/2 1/2 1/2 1 1 1 1 1];
order_x = 5;

Q = bsplineCurve(P3x, order_x, knots_x, 50);
hold on;
plot3(P3x(:,1),P3x(:,2), P3x(:,3),'r-s');
plot3(Q(:,1),Q(:,2), Q(:,3),'b','linewidth',2);
hold on;
axis equal;


a = P2x(1,:)-P1x(1,:);

CPs = [];
for j=1:size(P2x,1)
   
    tCP = P1x(:,1:3)+P2x(j,1:3); 
    disp(tCP);
    tCP(:,4) = P1x(:,4)*P2x(j,4);
    CPs = [CPs; tCP];
end

k1 = order_u;
k2 = order_v;
ncp_u = size(P1x,1);
ncp_v = size(P2x,1);

cph=[CPs(:,1:3).*CPs(:,4), CPs(:,4)];
cph;
CPs_d3 = reshape(cph',[4,ncp_u,ncp_v]);

nrbs_Spink.form='B-NURBS';
nrbs_Spink.dim = 4;
nrbs_Spink.number = [ncp_u, ncp_v];
nrbs_Spink.coefs = CPs_d3; 
nrbs_Spink.knots = {knots_u knots_v};
nrbs_Spink.order = [k1 k2];

%NURBS to IGES

%NrbsSrf2IGES(nrbs_Spink,'Sweep 6_13.igs','./');

nrbs.type   = 'Surface';
nrbs.number = [ncp_u, ncp_v];
nrbs.coefs = CPs_d3;  
nrbs.knots = {knots_u knots_v};
nrbs.order = [k1 k2];
%plotNrbs(nrbs);

