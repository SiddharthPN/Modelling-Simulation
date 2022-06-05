% Copyright: Xiaoping Qian @ UW-Madison
%% Right eye!
function righteye(Gen_Stl, DisplayCP)
    % profile 1
    D = [
        0.25 3.125;
        0.5 3.25;
        0.55 3.5;
        0.75 3.333;
        1.4 3.5;
        0.75 4.25;
        0.25 3.125;
        ];
    [rows, cols] = size(D);
    k=2;
    n=rows;

    % compute Bspline control points
    [P, knots_u] = BsplineInterpolate(D,k,n);
    x = [];
    w = [];
    x(1:rows,1) = 3.5;
    w(1:rows,1) = 1;
    P1 =[x,P,w];

    order_u = 2;

    % do plot of control polygon
    if DisplayCP == true
        Q = bsplineCurve(P1, order_u, knots_u, 50);
        plot3(P1(:,1),P1(:,2), P1(:,3),'r-s');
        hold on;
        plot3(Q(:,1),Q(:,2), Q(:,3),'b','linewidth',2);
        hold on;
    end


    % profile 2
    P2 = [-2, 0, 0, 0
        -2, 0, 0, 0];

    knots_v = [0 0 1 1];
    order_v = 2;

    %do plot of control polygon
    if DisplayCP == true
        Q = bsplineCurve(P2, order_v, knots_v, 50);
        plot3(P2(:,1),P2(:,2), P2(:,3),'r-s');
        hold on;
        plot3(Q(:,1),Q(:,2), Q(:,3),'b','linewidth',2);
        axis equal;
        xlabel('x');
        ylabel('y');
        zlabel('z');
    end

    [rows, cols] = size(P1);
    Cps = zeros(rows*2,cols);
    Cps(1:rows, 4) = P1(:,4);
    Cps(1:rows,1:3) = P1(:,1:3);
    j = 1; 

    for i=rows+1:1:2*rows
        Cps(i,1:4) = P1(j,1:4) + P2(2,:); 
        j = j + 1;
    end

    ncp_u = size(P1,1);
    ncp_v = size(P2,1);
    Cps = Cps';
    CPs_d3 = reshape(Cps,[4,ncp_u,ncp_v]);
    k1 = order_u;
    k2 = order_v; 
    nrbs.type   = 'Surface';
    nrbs.number = [ncp_u, 2];
    nrbs.coefs = CPs_d3;  % CPs in homogeneous coordinates
    nrbs.knots = {knots_u, knots_v};
    nrbs.order = [k1 k2];
    hold on;

    plotNrbs(nrbs);
    %print('-dpdf','-painters','revolution1.pdf')
    if Gen_Stl == true 
        fileStl = 'righteye.stl';
        [tri,vtx]=buildTriVtx(nrbs,[100,100]); % triangulate NURBS
        writeSTL(fileStl,'nW',tri,vtx,'mode','ascii'); % write STL
    %     [vtx,tri]=readSTL(fileStl,'y');% read STL file
    end
end