% Pumpkin Stem Extrusion
function Pumpkin_connect(Gen_Stl, DisplayCP)    
    load Stem_Control_Points.mat;

    P1 = [
        -0.3878, 0.4862;
        -0.4579, 0.7877;
        -0.1323, 0.6145;
        0.0711, 0.5819;
        0.3122, 0.7473;
        0.3623, 0.4785;
        0.4827, 0.3547;
        0.7478, 0.3078;
        0.5923, 0.0703;
        0.5896, -0.0768;
        0.7419, -0.3070;
        0.4898, -0.3701;
        0.3505, -0.4764;
        0.3091, -0.7499;
        0.0694, -0.5906;
        -0.0836, -0.5906;
        -0.3065, -0.7523;
        -0.3779, -0.4517;
        -0.4662, -0.3636;
        -0.7486, -0.3243;
        -0.5785, -0.0493;
        -0.6271, 0.1610;
        -0.8167, 0.5918;
        -0.3878, 0.4862;
        ];
    [rows, cols] = size(P1);
    k=2;
    n=rows;

    % compute Bspline control points
    knots_u = [0,0,0,0,0.0898623442043661,0.133524492994021,0.174690140567622,0.214686389691028,0.254858946777618,0.295125479976933,0.336543965232339,0.377771753079734,0.418919546379191,0.458947913143236,0.499188171337169,0.539525215671418,0.580979808281318,0.623167497148064,0.665117756899622,0.708675086855341,0.752931570705175,0.796752617407235,0.839587582455233,0.887180578854278,1,1,1,1];
    z = [];
    w = [];
    x(1:rows,1) = 4.5780;
    w(1:rows,1) = 1;
    P1 =[P1,x,w];

    order_u = 4;

    % do plot of control polygon
    if DisplayCP == true
        Q = bsplineCurve(P1, order_u, knots_u, 50);
        plot3(P1(:,1),P1(:,2), P1(:,3),'r-s');
        hold on;
        plot3(Q(:,1),Q(:,2), Q(:,3),'b','linewidth',2);
        hold on;
    end
    % profile 2
    P2 = [0, 0, 0, 0
        0, 0, -0.25, 0];

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

    plotNrbs(nrbs);
    hold on;

    %print('-dpdf','-painters','revolution1.pdf')
    if Gen_Stl == true 
        fileStl = 'Stem_connect.stl';
        [tri,vtx]=buildTriVtx(nrbs,[100,100]); % triangulate NURBS
        writeSTL(fileStl,'y',tri,vtx,'mode','ascii'); % write STL
        [vtx,tri]=readSTL(fileStl,'n');% read STL file
    end
end
