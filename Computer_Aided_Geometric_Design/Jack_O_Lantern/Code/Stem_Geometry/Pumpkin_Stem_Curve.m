% Pumpkin Stem
% input data point, order of the curve

% Function call


[P1,knots1] = Interpolate1(0,0);
[P2,knots2] = Interpolate1(0.05,0.05);
[P3,knots3] = Interpolate2(-0.09,0.3);
[P4,knots4] = Interpolate2(-0.08,0.4);
[P5,knots5] = Interpolate2(-0.07,0.5);
[P6,knots6] = Interpolate2(-0.06,0.6);
[P7,knots7] = Interpolate2(-0.05,0.7);
[P8,knots8] = Interpolate2(-0.04,0.8);
[P9,knots9] = Interpolate2(-0.03,0.9);
[P10,knots10] = Interpolate2(-0.02,1.0);
[P11,knots11] = Interpolate2(-0.01,1.1);
[P12,knots12] = Interpolate2(0,1.2);
[P13,knots13] = Interpolate3(0.05,1.4);
[P14,knots14] = Interpolate3(0.07,1.6);
[P15,knots15] = Interpolate4(0.07,6.4);
[P16,knots16] = Interpolate5(0.1,6.4,0.5);
[P17,knots17] = Interpolate6(0.1,6.4,0.8);
[P18,knots18] = Interpolate7(0.1,6.2,1.1);
[P19,knots19] = Interpolate8(0.1,5.8,1.3);


xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis');

function [P,knots] = Interpolate8(d,z,m)

% Original Data points for Stem Base

Q1a = [-0.121948+d     0.295667-d    4.36662];
Q1b = [-0.178389+d     0.223692-d    4.36662];
Q1c = [-0.028044+d,    0.284736-d    4.36662];
Q1a([2 3]) = Q1a([3 2]);
Q1b([2 3]) = Q1b([3 2]);
Q1c([2 3]) = Q1c([3 2]);

Q2a = [0.122837-d      0.295298-d    4.36662];
Q2b = [0.0320345-d     0.284314-d    4.36662];
Q2c = [0.181508-d      0.221169-d    4.36662];
Q2a([2 3]) = Q2a([3 2]);
Q2b([2 3]) = Q2b([3 2]);
Q2c([2 3]) = Q2c([3 2]);

Q3a = [0.295667-d      0.121948-d    4.36662];
Q3b = [0.223692-d      0.178389-d    4.36662];
Q3c = [0.284736-d      0.028044-d    4.36662];
Q3a([2 3]) = Q3a([3 2]);
Q3b([2 3]) = Q3b([3 2]);
Q3c([2 3]) = Q3c([3 2]);

Q4a = [0.295298-d     -0.122837+d    4.36662];
Q4b = [0.284314-d     -0.0320345+d   4.36662];
Q4c = [0.220375-d     -0.184612+d    4.36662];
Q4a([2 3]) = Q4a([3 2]);
Q4b([2 3]) = Q4b([3 2]);
Q4c([2 3]) = Q4c([3 2]);

Q5a = [0.121948-d     -0.295667+d    4.36662];
Q5b = [0.178389-d     -0.223692+d    4.36662];
Q5c = [0.028044-d     -0.284736+d    4.36662];
Q5a([2 3]) = Q5a([3 2]);
Q5b([2 3]) = Q5b([3 2]);
Q5c([2 3]) = Q5c([3 2]);

Q6a = [-0.122837+d    -0.295298+d    4.36662];
Q6b = [-0.0320345+d   -0.284314+d    4.36662];
Q6c = [-0.181508+d    -0.221169+d    4.36662];
Q6a([2 3]) = Q6a([3 2]);
Q6b([2 3]) = Q6b([3 2]);
Q6c([2 3]) = Q6c([3 2]);

Q7a = [-0.295667+d    -0.121948+d    4.36662];
Q7b = [-0.223692+d    -0.178389+d    4.36662];
Q7c = [-0.284736+d    -0.028044+d    4.36662];
Q7a([2 3]) = Q7a([3 2]);
Q7b([2 3]) = Q7b([3 2]);
Q7c([2 3]) = Q7c([3 2]);

Q8a = [-0.295298+d     0.122837-d    4.36662];
Q8b = [-0.284314+d     0.0320345-d   4.36662];
Q8c = [-0.221169+d     0.181508-d    4.36662];
Q8a([2 3]) = Q8a([3 2]);
Q8b([2 3]) = Q8b([3 2]);
Q8c([2 3]) = Q8c([3 2]);


%Data point array
D = [Q1b/5; Q2b/5; Q1a/5; 
     Q2a/5; Q1c/5; Q2c/5;  
     Q3b/5; Q4b/5; Q3a/5; 
     Q4a/5; Q3c/5; Q4c/5; 
     Q5b/5; Q6b/5; Q5a/5; 
     Q6a/5; Q5c/5; Q6c/5; 
     Q7b/5; Q8b/5; Q7a/5;
     Q8a/5; Q7c/5; Q1b/5];

%Height of curve
D(:,3) = D(:,3) + z;
D(:,2) = m;

k=3;
n=24;

% compute Bspline control points
[P, knots] = BsplineInterpolate(D,k,n);

% plot data points
plot3(D(:,1), D(:,2), D(:,3),'rs');
hold on;

%define display configuration
dn = 40; % # of points on the B-spline curve
%do calculation
Q = bsplineCurve(P, k, knots, dn);

%do plot of control polygon
%plot3(P(:,1),P(:,2),P(:,3),'m-o');
hold on;
%do plot of b-spline curve
plot3(Q(:,1),Q(:,2),Q(:,3));
title('B-spline Curve');
grid on;
hold on;
end

function [P,knots] = Interpolate7(d,z,m)

% Original Data points for Stem Base

Q1a = [-0.121948+d     0.295667-d    4.36662];
Q1b = [-0.178389+d     0.223692-d    4.36662];
Q1c = [-0.028044+d,    0.284736-d    4.36662];
Q1a([2 3]) = Q1a([3 2]);
Q1b([2 3]) = Q1b([3 2]);
Q1c([2 3]) = Q1c([3 2]);

Q2a = [0.122837-d      0.295298-d    4.36662];
Q2b = [0.0320345-d     0.284314-d    4.36662];
Q2c = [0.181508-d      0.221169-d    4.36662];
Q2a([2 3]) = Q2a([3 2]);
Q2b([2 3]) = Q2b([3 2]);
Q2c([2 3]) = Q2c([3 2]);

Q3a = [0.295667-d      0.121948-d    4.36662];
Q3b = [0.223692-d      0.178389-d    4.36662];
Q3c = [0.284736-d      0.028044-d    4.36662];
Q3a([2 3]) = Q3a([3 2]);
Q3b([2 3]) = Q3b([3 2]);
Q3c([2 3]) = Q3c([3 2]);

Q4a = [0.295298-d     -0.122837+d    4.36662];
Q4b = [0.284314-d     -0.0320345+d   4.36662];
Q4c = [0.220375-d     -0.184612+d    4.36662];
Q4a([2 3]) = Q4a([3 2]);
Q4b([2 3]) = Q4b([3 2]);
Q4c([2 3]) = Q4c([3 2]);

Q5a = [0.121948-d     -0.295667+d    4.36662];
Q5b = [0.178389-d     -0.223692+d    4.36662];
Q5c = [0.028044-d     -0.284736+d    4.36662];
Q5a([2 3]) = Q5a([3 2]);
Q5b([2 3]) = Q5b([3 2]);
Q5c([2 3]) = Q5c([3 2]);

Q6a = [-0.122837+d    -0.295298+d    4.36662];
Q6b = [-0.0320345+d   -0.284314+d    4.36662];
Q6c = [-0.181508+d    -0.221169+d    4.36662];
Q6a([2 3]) = Q6a([3 2]);
Q6b([2 3]) = Q6b([3 2]);
Q6c([2 3]) = Q6c([3 2]);

Q7a = [-0.295667+d    -0.121948+d    4.36662];
Q7b = [-0.223692+d    -0.178389+d    4.36662];
Q7c = [-0.284736+d    -0.028044+d    4.36662];
Q7a([2 3]) = Q7a([3 2]);
Q7b([2 3]) = Q7b([3 2]);
Q7c([2 3]) = Q7c([3 2]);

Q8a = [-0.295298+d     0.122837-d    4.36662];
Q8b = [-0.284314+d     0.0320345-d   4.36662];
Q8c = [-0.221169+d     0.181508-d    4.36662];
Q8a([2 3]) = Q8a([3 2]);
Q8b([2 3]) = Q8b([3 2]);
Q8c([2 3]) = Q8c([3 2]);


%Data point array
D = [Q1b/3.5; Q2b/3.5; Q1a/3.5; 
     Q2a/3.5; Q1c/3.5; Q2c/3.5;  
     Q3b/3.5; Q4b/3.5; Q3a/3.5; 
     Q4a/3.5; Q3c/3.5; Q4c/3.5; 
     Q5b/3.5; Q6b/3.5; Q5a/3.5; 
     Q6a/3.5; Q5c/3.5; Q6c/3.5; 
     Q7b/3.5; Q8b/3.5; Q7a/3.5;
     Q8a/3.5; Q7c/3.5; Q1b/3.5];

%Height of curve
D(:,3) = D(:,3) + z;
D(:,2) = m;

k=3;
n=24;

% compute Bspline control points
[P, knots] = BsplineInterpolate(D,k,n);

% plot data points
plot3(D(:,1), D(:,2), D(:,3),'rs');
hold on;

%define display configuration
dn = 40; % # of points on the B-spline curve
%do calculation
Q = bsplineCurve(P, k, knots, dn);

%do plot of control polygon
%plot3(P(:,1),P(:,2),P(:,3),'m-o');
hold on;
%do plot of b-spline curve
plot3(Q(:,1),Q(:,2),Q(:,3));
title('B-spline Curve');
grid on;
hold on;
end

function [P,knots] = Interpolate6(d,z,m)

% Original Data points for Stem Base

Q1a = [-0.121948+d     0.295667-d    4.36662];
Q1b = [-0.178389+d     0.223692-d    4.36662];
Q1c = [-0.028044+d,    0.284736-d    4.36662];
Q1a([2 3]) = Q1a([3 2]);
Q1b([2 3]) = Q1b([3 2]);
Q1c([2 3]) = Q1c([3 2]);

Q2a = [0.122837-d      0.295298-d    4.36662];
Q2b = [0.0320345-d     0.284314-d    4.36662];
Q2c = [0.181508-d      0.221169-d    4.36662];
Q2a([2 3]) = Q2a([3 2]);
Q2b([2 3]) = Q2b([3 2]);
Q2c([2 3]) = Q2c([3 2]);

Q3a = [0.295667-d      0.121948-d    4.36662];
Q3b = [0.223692-d      0.178389-d    4.36662];
Q3c = [0.284736-d      0.028044-d    4.36662];
Q3a([2 3]) = Q3a([3 2]);
Q3b([2 3]) = Q3b([3 2]);
Q3c([2 3]) = Q3c([3 2]);

Q4a = [0.295298-d     -0.122837+d    4.36662];
Q4b = [0.284314-d     -0.0320345+d   4.36662];
Q4c = [0.220375-d     -0.184612+d    4.36662];
Q4a([2 3]) = Q4a([3 2]);
Q4b([2 3]) = Q4b([3 2]);
Q4c([2 3]) = Q4c([3 2]);

Q5a = [0.121948-d     -0.295667+d    4.36662];
Q5b = [0.178389-d     -0.223692+d    4.36662];
Q5c = [0.028044-d     -0.284736+d    4.36662];
Q5a([2 3]) = Q5a([3 2]);
Q5b([2 3]) = Q5b([3 2]);
Q5c([2 3]) = Q5c([3 2]);

Q6a = [-0.122837+d    -0.295298+d    4.36662];
Q6b = [-0.0320345+d   -0.284314+d    4.36662];
Q6c = [-0.181508+d    -0.221169+d    4.36662];
Q6a([2 3]) = Q6a([3 2]);
Q6b([2 3]) = Q6b([3 2]);
Q6c([2 3]) = Q6c([3 2]);

Q7a = [-0.295667+d    -0.121948+d    4.36662];
Q7b = [-0.223692+d    -0.178389+d    4.36662];
Q7c = [-0.284736+d    -0.028044+d    4.36662];
Q7a([2 3]) = Q7a([3 2]);
Q7b([2 3]) = Q7b([3 2]);
Q7c([2 3]) = Q7c([3 2]);

Q8a = [-0.295298+d     0.122837-d    4.36662];
Q8b = [-0.284314+d     0.0320345-d   4.36662];
Q8c = [-0.221169+d     0.181508-d    4.36662];
Q8a([2 3]) = Q8a([3 2]);
Q8b([2 3]) = Q8b([3 2]);
Q8c([2 3]) = Q8c([3 2]);


%Data point array
D = [Q1b/2.5; Q2b/2.5; Q1a/2.5; 
     Q2a/2.5; Q1c/2.5; Q2c/2.5;  
     Q3b/2.5; Q4b/2.5; Q3a/2.5; 
     Q4a/2.5; Q3c/2.5; Q4c/2.5; 
     Q5b/2.5; Q6b/2.5; Q5a/2.5; 
     Q6a/2.5; Q5c/2.5; Q6c/2.5; 
     Q7b/2.5; Q8b/2.5; Q7a/2.5;
     Q8a/2.5; Q7c/2.5; Q1b/2.5];

%Height of curve
D(:,3) = D(:,3) + z;
D(:,2) = m;

k=3;
n=24;

% compute Bspline control points
[P, knots] = BsplineInterpolate(D,k,n);

% plot data points
plot3(D(:,1), D(:,2), D(:,3),'rs');
hold on;

%define display configuration
dn = 40; % # of points on the B-spline curve
%do calculation
Q = bsplineCurve(P, k, knots, dn);

%do plot of control polygon
%plot3(P(:,1),P(:,2),P(:,3),'m-o');
hold on;
%do plot of b-spline curve
plot3(Q(:,1),Q(:,2),Q(:,3));
title('B-spline Curve');
grid on;
hold on;
end

function [P,knots] = Interpolate5(d,z,m)

% Original Data points for Stem Base

Q1a = [-0.121948+d     0.295667-d    4.36662];
Q1b = [-0.178389+d     0.223692-d    4.36662];
Q1c = [-0.028044+d,    0.284736-d    4.36662];
Q1a([2 3]) = Q1a([3 2]);
Q1b([2 3]) = Q1b([3 2]);
Q1c([2 3]) = Q1c([3 2]);

Q2a = [0.122837-d      0.295298-d    4.36662];
Q2b = [0.0320345-d     0.284314-d    4.36662];
Q2c = [0.181508-d      0.221169-d    4.36662];
Q2a([2 3]) = Q2a([3 2]);
Q2b([2 3]) = Q2b([3 2]);
Q2c([2 3]) = Q2c([3 2]);

Q3a = [0.295667-d      0.121948-d    4.36662];
Q3b = [0.223692-d      0.178389-d    4.36662];
Q3c = [0.284736-d      0.028044-d    4.36662];
Q3a([2 3]) = Q3a([3 2]);
Q3b([2 3]) = Q3b([3 2]);
Q3c([2 3]) = Q3c([3 2]);

Q4a = [0.295298-d     -0.122837+d    4.36662];
Q4b = [0.284314-d     -0.0320345+d   4.36662];
Q4c = [0.220375-d     -0.184612+d    4.36662];
Q4a([2 3]) = Q4a([3 2]);
Q4b([2 3]) = Q4b([3 2]);
Q4c([2 3]) = Q4c([3 2]);

Q5a = [0.121948-d     -0.295667+d    4.36662];
Q5b = [0.178389-d     -0.223692+d    4.36662];
Q5c = [0.028044-d     -0.284736+d    4.36662];
Q5a([2 3]) = Q5a([3 2]);
Q5b([2 3]) = Q5b([3 2]);
Q5c([2 3]) = Q5c([3 2]);

Q6a = [-0.122837+d    -0.295298+d    4.36662];
Q6b = [-0.0320345+d   -0.284314+d    4.36662];
Q6c = [-0.181508+d    -0.221169+d    4.36662];
Q6a([2 3]) = Q6a([3 2]);
Q6b([2 3]) = Q6b([3 2]);
Q6c([2 3]) = Q6c([3 2]);

Q7a = [-0.295667+d    -0.121948+d    4.36662];
Q7b = [-0.223692+d    -0.178389+d    4.36662];
Q7c = [-0.284736+d    -0.028044+d    4.36662];
Q7a([2 3]) = Q7a([3 2]);
Q7b([2 3]) = Q7b([3 2]);
Q7c([2 3]) = Q7c([3 2]);

Q8a = [-0.295298+d     0.122837-d    4.36662];
Q8b = [-0.284314+d     0.0320345-d   4.36662];
Q8c = [-0.221169+d     0.181508-d    4.36662];
Q8a([2 3]) = Q8a([3 2]);
Q8b([2 3]) = Q8b([3 2]);
Q8c([2 3]) = Q8c([3 2]);


%Data point array
D = [Q1b/1.5; Q2b/1.5; Q1a/1.5; 
     Q2a/1.5; Q1c/1.5; Q2c/1.5;  
     Q3b/1.5; Q4b/1.5; Q3a/1.5; 
     Q4a/1.5; Q3c/1.5; Q4c/1.5; 
     Q5b/1.5; Q6b/1.5; Q5a/1.5; 
     Q6a/1.5; Q5c/1.5; Q6c/1.5; 
     Q7b/1.5; Q8b/1.5; Q7a/1.5;
     Q8a/1.5; Q7c/1.5; Q1b/1.5];

%Height of curve
D(:,3) = D(:,3) + z;
D(:,2) = m;

k=3;
n=24;

% compute Bspline control points
[P, knots] = BsplineInterpolate(D,k,n);

% plot data points
plot3(D(:,1), D(:,2), D(:,3),'rs');
hold on;

%define display configuration
dn = 40; % # of points on the B-spline curve
%do calculation
Q = bsplineCurve(P, k, knots, dn);

%do plot of control polygon
%plot3(P(:,1),P(:,2),P(:,3),'m-o');
hold on;
%do plot of b-spline curve
plot3(Q(:,1),Q(:,2),Q(:,3));
title('B-spline Curve');
grid on;
hold on;
end

function [P,knots] = Interpolate4(d,z)

% Original Data points for Stem Base

Q1a = [-0.121948+d     0.295667-d    4.36662];
Q1b = [-0.178389+d     0.223692-d    4.36662];
Q1c = [-0.028044+d,    0.284736-d    4.36662];
Q1a([2 3]) = Q1a([3 2]);
Q1b([2 3]) = Q1b([3 2]);
Q1c([2 3]) = Q1c([3 2]);

Q2a = [0.122837-d      0.295298-d    4.36662];
Q2b = [0.0320345-d     0.284314-d    4.36662];
Q2c = [0.181508-d      0.221169-d    4.36662];
Q2a([2 3]) = Q2a([3 2]);
Q2b([2 3]) = Q2b([3 2]);
Q2c([2 3]) = Q2c([3 2]);

Q3a = [0.295667-d      0.121948-d    4.36662];
Q3b = [0.223692-d      0.178389-d    4.36662];
Q3c = [0.284736-d      0.028044-d    4.36662];
Q3a([2 3]) = Q3a([3 2]);
Q3b([2 3]) = Q3b([3 2]);
Q3c([2 3]) = Q3c([3 2]);

Q4a = [0.295298-d     -0.122837+d    4.36662];
Q4b = [0.284314-d     -0.0320345+d   4.36662];
Q4c = [0.220375-d     -0.184612+d    4.36662];
Q4a([2 3]) = Q4a([3 2]);
Q4b([2 3]) = Q4b([3 2]);
Q4c([2 3]) = Q4c([3 2]);

Q5a = [0.121948-d     -0.295667+d    4.36662];
Q5b = [0.178389-d     -0.223692+d    4.36662];
Q5c = [0.028044-d     -0.284736+d    4.36662];
Q5a([2 3]) = Q5a([3 2]);
Q5b([2 3]) = Q5b([3 2]);
Q5c([2 3]) = Q5c([3 2]);

Q6a = [-0.122837+d    -0.295298+d    4.36662];
Q6b = [-0.0320345+d   -0.284314+d    4.36662];
Q6c = [-0.181508+d    -0.221169+d    4.36662];
Q6a([2 3]) = Q6a([3 2]);
Q6b([2 3]) = Q6b([3 2]);
Q6c([2 3]) = Q6c([3 2]);

Q7a = [-0.295667+d    -0.121948+d    4.36662];
Q7b = [-0.223692+d    -0.178389+d    4.36662];
Q7c = [-0.284736+d    -0.028044+d    4.36662];
Q7a([2 3]) = Q7a([3 2]);
Q7b([2 3]) = Q7b([3 2]);
Q7c([2 3]) = Q7c([3 2]);

Q8a = [-0.295298+d     0.122837-d    4.36662];
Q8b = [-0.284314+d     0.0320345-d   4.36662];
Q8c = [-0.221169+d     0.181508-d    4.36662];
Q8a([2 3]) = Q8a([3 2]);
Q8b([2 3]) = Q8b([3 2]);
Q8c([2 3]) = Q8c([3 2]);


%Data point array
D = [Q1b; Q1a; Q2b; 
     Q1c; Q2a; Q2c;  
     Q3b; Q3a; Q4b; 
     Q3c; Q4a; Q4c; 
     Q5b; Q5a; Q6b; 
     Q5c; Q6a; Q6c; 
     Q7b; Q7a; Q8b;
     Q7c; Q8a; Q1b];

%Height of curve
D(:,3) = D(:,3) + z;
D(:,2) = 0.3;
D4 = D;
D4

k=3;
n=24;

% compute Bspline control points
[P, knots] = BsplineInterpolate(D,k,n);

% plot data points
plot3(D(:,1), D(:,2), D(:,3),'rs');
hold on;

%define display configuration
dn = 40; % # of points on the B-spline curve
%do calculation
Q = bsplineCurve(P, k, knots, dn);

%do plot of control polygon
%plot3(P(:,1),P(:,2),P(:,3),'m-o');
hold on;
%do plot of b-spline curve
plot3(Q(:,1),Q(:,2),Q(:,3));
title('B-spline Curve');
grid on;
hold on;
end


function [P,knots] = Interpolate3(d,z)

% Original Data points for Stem Base

Q1a = [-0.121948+d     0.295667-d    4.36662];
Q1b = [-0.178389+d     0.223692-d    4.36662];
Q1c = [-0.028044+d,    0.284736-d    4.36662];

Q2a = [0.122837-d      0.295298-d    4.36662];
Q2b = [0.0320345-d     0.284314-d    4.36662];
Q2c = [0.181508-d      0.221169-d    4.36662];

Q3a = [0.295667-d      0.121948-d    4.36662];
Q3b = [0.223692-d      0.178389-d    4.36662];
Q3c = [0.284736-d      0.028044-d    4.36662];

Q4a = [0.295298-d     -0.122837+d    4.36662];
Q4b = [0.284314-d     -0.0320345+d   4.36662];
Q4c = [0.220375-d     -0.184612+d    4.36662];

Q5a = [0.121948-d     -0.295667+d    4.36662];
Q5b = [0.178389-d     -0.223692+d    4.36662];
Q5c = [0.028044-d     -0.284736+d    4.36662];

Q6a = [-0.122837+d    -0.295298+d    4.36662];
Q6b = [-0.0320345+d   -0.284314+d    4.36662];
Q6c = [-0.181508+d    -0.221169+d    4.36662];

Q7a = [-0.295667+d    -0.121948+d    4.36662];
Q7b = [-0.223692+d    -0.178389+d    4.36662];
Q7c = [-0.284736+d    -0.028044+d    4.36662];

Q8a = [-0.295298+d     0.122837-d    4.36662];
Q8b = [-0.284314+d     0.0320345-d   4.36662];
Q8c = [-0.221169+d     0.181508-d    4.36662];


%Data point array
D = [Q1b; Q1a; Q2b; 
     Q1c; Q2a; Q2c;  
     Q3b; Q3a; Q4b; 
     Q3c; Q4a; Q4c; 
     Q5b; Q5a; Q6b; 
     Q5c; Q6a; Q6c; 
     Q7b; Q7a; Q8b;
     Q7c; Q8a; Q1b];

%Height of curve
D(:,3) = D(:,3) + z;

D3 = D;
D3

k=3;
n=24;

% compute Bspline control points
[P, knots] = BsplineInterpolate(D,k,n);

% plot data points
plot3(D(:,1), D(:,2), D(:,3),'rs');
hold on;

%define display configuration
dn = 40; % # of points on the B-spline curve
%do calculation
Q = bsplineCurve(P, k, knots, dn);

%do plot of control polygon
%plot3(P(:,1),P(:,2),P(:,3),'m-o');
hold on;
%do plot of b-spline curve
plot3(Q(:,1),Q(:,2),Q(:,3));
title('B-spline Curve');
grid on;
hold on;
end



function [P,knots] = Interpolate2(d,z)

% Original Data points for Stem Base

Q1a = [-0.121948+d     0.295667-d    4.36662];
Q1b = [-0.178389+d     0.223692-d    4.36662];
Q1c = [-0.028044+d,    0.284736-d    4.36662];

Q2a = [0.122837-d      0.295298-d    4.36662];
Q2b = [0.0320345-d     0.284314-d    4.36662];
Q2c = [0.181508-d      0.221169-d    4.36662];

Q3a = [0.295667-d      0.121948-d    4.36662];
Q3b = [0.223692-d      0.178389-d    4.36662];
Q3c = [0.284736-d      0.028044-d    4.36662];

Q4a = [0.295298-d     -0.122837+d    4.36662];
Q4b = [0.284314-d     -0.0320345+d   4.36662];
Q4c = [0.220375-d     -0.184612+d    4.36662];

Q5a = [0.121948-d     -0.295667+d    4.36662];
Q5b = [0.178389-d     -0.223692+d    4.36662];
Q5c = [0.028044-d     -0.284736+d    4.36662];

Q6a = [-0.122837+d    -0.295298+d    4.36662];
Q6b = [-0.0320345+d   -0.284314+d    4.36662];
Q6c = [-0.181508+d    -0.221169+d    4.36662];

Q7a = [-0.295667+d    -0.121948+d    4.36662];
Q7b = [-0.223692+d    -0.178389+d    4.36662];
Q7c = [-0.284736+d    -0.028044+d    4.36662];

Q8a = [-0.295298+d     0.122837-d    4.36662];
Q8b = [-0.284314+d     0.0320345-d   4.36662];
Q8c = [-0.221169+d     0.181508-d    4.36662];


%Data point array
D = [Q1b; Q1a; Q1c; 
     Q2b; Q2a; Q2c;  
     Q3b; Q3a; Q3c; 
     Q4b; Q4a; Q4c; 
     Q5b; Q5a; Q5c; 
     Q6b; Q6a; Q6c; 
     Q7b; Q7a; Q7c;
     Q8b; Q8a; Q1b];

%Height of curve
D(:,3) = D(:,3) + z;

k=3;
n=24;

% compute Bspline control points
[P, knots] = BsplineInterpolate(D,k,n);

% plot data points
plot3(D(:,1), D(:,2), D(:,3),'rs');
hold on;

%define display configuration
dn = 40; % # of points on the B-spline curve
%do calculation
Q = bsplineCurve(P, k, knots, dn);

%do plot of control polygon
%plot3(P(:,1),P(:,2),P(:,3),'m-o');
hold on;
%do plot of b-spline curve
plot3(Q(:,1),Q(:,2),Q(:,3));
title('B-spline Curve');
grid on;
hold on;
end



function [P,knots] = Interpolate1(d,z)

% Original Data points for Stem Base

Q1a = [-0.265078+d     0.642687-d    4.578];
Q1b = [-0.387762+d     0.486238-d    4.578];
Q1c = [-0.0609589+d,   0.618926-d    4.578];
Q1d = [4.4802e-17      0.731672-d    4.578];

Q2a = [0.26701-d       0.641887-d    4.578];
Q2b = [0.0696331-d     0.618011-d    4.578];
Q2c = [0.394543-d      0.480752-d    4.578];
Q2d = [0.51737-d       0.51737-d     4.578];

Q3a = [0.642687-d      0.265078-d    4.578];
Q3b = [0.486238-d      0.387762-d    4.578];
Q3c = [0.618926-d      0.0609589-d   4.578];
Q3d = [0.731672-d     -2.22045e-16   4.578];

Q4a = [0.641887-d     -0.26701+d     4.578];
Q4b = [0.618011-d     -0.0696331+d   4.578];
Q4c = [0.48752-d      -0.394543+d    4.578];
Q4d = [0.51737-d      -0.51737+d     4.578];

Q5a = [0.265078-d     -0.642687+d    4.578];
Q5b = [0.387762-d     -0.486238+d    4.578];
Q5c = [0.0609589-d    -0.618926+d    4.578];
Q5d = [-1.34406e-16   -0.731672+d    4.578];

Q6a = [-0.26701+d     -0.641887+d    4.578];
Q6b = [-0.0696331+d   -0.618011+d    4.578];
Q6c = [-0.394543+d    -0.480752+d    4.578];
Q6d = [-0.51737+d     -0.51737+d     4.578];

Q7a = [-0.642687+d    -0.265078+d    4.578];
Q7b = [-0.486238+d    -0.387762+d    4.578];
Q7c = [-0.618926+d    -0.0609589+d   4.578];
Q7d = [-0.731672+d     1.3145e-16    4.578];

Q8a = [-0.641887+d     0.26701-d     4.578];
Q8b = [-0.618011+d     0.0696331-d   4.578];
Q8c = [-0.480752+d     0.394543-d    4.578];
Q8d = [-0.51737+d      0.51737-d     4.578];


%Data point array
D = [Q1b; Q1a; Q1c; 
     Q2b; Q2a; Q2c;  
     Q3b; Q3a; Q3c; 
     Q4b; Q4a; Q4c; 
     Q5b; Q5a; Q5c; 
     Q6b; Q6a; Q6c; 
     Q7b; Q7a; Q7c;
     Q8b; Q8a; Q1b];

%Height of curve
D(:,3) = D(:,3) + z;


k=3;
n=24;

% compute Bspline control points
[P, knots] = BsplineInterpolate(D,k,n);

% plot data points
plot3(D(:,1), D(:,2), D(:,3),'rs');
hold on;

%define display configuration
dn = 40; % # of points on the B-spline curve
%do calculation
Q = bsplineCurve(P, k, knots, dn);

%do plot of control polygon
%plot3(P(:,1),P(:,2),P(:,3),'m-o');
hold on;
%do plot of b-spline curve
plot3(Q(:,1),Q(:,2),Q(:,3));
title('B-spline Curve');
grid on;
hold on;
end

