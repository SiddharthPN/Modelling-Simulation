function [Px,Py,Pz,U,V] = BsplineSurfInter(Dx,Dy,Dz,ku,kv,nu,nv)
%B-spline surface interpolation,
% this is a global interpolation routine
% input: Dx, Dy, Dz (grid data point coordinates x,y correspondingly), 
%       nu,nv (n=# of data point, u,v direction correspondingly)
%       ku,kv (order of the curve, u,v direction correspondingly), 
% output: Px,Py,Pz (control point of the resulting B-spline surface, coordinates x,y,z correspondingly)
%       U, V knot vectors in u and v directions.

%test data;
% Dx=[0 3 4 5 8 10 8 0;2 3 3 5 3 6 8 1;-1 2 3 5 3 9 7 0;-2 3 4 5 6 2 5 7]; % data points
% Dy=[2 3 5 4 10 1 3 0;1 2 3 6 4 5 2 9;-2 3 4 5 6 2 5 7;-1 2 3 5 3 9 7 0];
% ku=2;
% kv=2;
% nu=8;
% nv=4;

% Step 0:
% Number of data points should be no less than the Order of the surface
if(ku>nu) 
    ku = nu;
end

if(kv>nv) 
    kv = nv;
end

% Step 1: Computing parameters: ubu(), ubv()
ubu = zeros(1,nu);
for l=0:nv-1,
    totalChord = 0;
    for k=1:nu-1,
        cds(k) = sqrt((Dx(l+1,k+1)-Dx(l+1,k))^2+(Dy(l+1,k+1)-Dy(l+1,k))^2+(Dz(l+1,k+1)-Dz(l+1,k))^2);
        totalChord = totalChord + cds(k);
    end
    
    dt = 0;   
    for k=1:nu-1,
        dt = dt + cds(k);
        ubu(k+1) = ubu(k+1) + dt/totalChord;
    end    
end

ubu(1) = 0.;
for k=1:nu-2,
    ubu(k+1) = ubu(k+1)/nv;
end
ubu(nu) = 1.0 ;
%%%%%%%%%%%%%%%%%%%%%%%%%%% Para v direction
ubv = zeros(1,nv);
for k=0:nu-1,
    totalChord = 0;
    for l=1:nv-1,
        cds(l) = sqrt((Dx(l+1,k+1)-Dx(l,k+1))^2+(Dy(l+1,k+1)-Dy(l,k+1))^2+(Dz(l+1,k+1)-Dz(l,k+1))^2);
        totalChord = totalChord + cds(l);
    end
    
    dt = 0;   
    for l=1:nv-1,
        dt = dt + cds(l);
        ubv(l+1) = ubv(l+1) + dt/totalChord;
    end    
end

ubv(1) = 0.;
for l=1:nv-2,
    ubv(l+1) = ubv(l+1)/nu;
end
ubv(nv) = 1.0 ;

%Step 2: Computing knot vector: U,V
for j=1:nu-ku,
    U(j+ku) = 0.;
    for i=j:j+ku-2 % averaging k-1 parameters
        U(j+ku) = U(j+ku)+ubu(i+1);
    end
    U(j+ku) = U(j+ku)/(ku-1);
end

for j=1:ku,
    U(j) = 0.; % repeating 1st k knots
end

for j=nu+1:nu+ku,
    U(j) = 1.; % repeating last k knots
end
%%%%%%%%%%%%%%%%%%%%%%Calculate V
for j=1:nv-kv,
    V(j+kv) = 0.;
    for i=j:j+kv-2 % averaging k-1 parameters
        V(j+kv) = V(j+kv)+ubv(i+1);
    end
    V(j+kv) = V(j+kv)/(kv-1);
end

for j=1:kv,
    V(j) = 0.; % repeating 1st k knots
end

for j=nv+1:nv+kv,
    V(j) = 1.; % repeating last k knots
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 3: Get the intermediate control points Q

for i=0:nv-1,
    Qx(i+1,:) = (InterForSurf(Dx(i+1,:)',ku,nu,ubu,U))';
    Qy(i+1,:) = (InterForSurf(Dy(i+1,:)',ku,nu,ubu,U))';
    Qz(i+1,:) = (InterForSurf(Dz(i+1,:)',ku,nu,ubu,U))';
end

% hold on;
% plot3(Qx',Qy',Qz','g-o', 'MarkerFaceColor', 'g', 'MarkerSize',8);
% axis off;
% view(3);

%Step 4: Get the final Control points P

for j=0:nu-1,
    Px(:,j+1) = InterForSurf(Qx(:,j+1),kv,nv,ubv,V);
    Py(:,j+1) = InterForSurf(Qy(:,j+1),kv,nv,ubv,V);
    Pz(:,j+1) = InterForSurf(Qz(:,j+1),kv,nv,ubv,V);
end

%plot3(Px,Py,Pz,'b-s', 'MarkerFaceColor', 'b', 'MarkerSize',8);
%plot3(Qx',Qy',Qz','go', 'MarkerFaceColor', 'g', 'MarkerSize',8);
axis off;
view(3);

%Step 5: End of program







