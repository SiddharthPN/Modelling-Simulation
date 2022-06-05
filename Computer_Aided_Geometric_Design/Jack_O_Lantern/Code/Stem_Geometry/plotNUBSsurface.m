function plotNUBSsurface(p,q,U,V,CP,num)

% PLOTNUBSSurface generates a polygonal surface that approximates a
% nonuniform b-spline surface of degree (p,q) given by the control
% net CP and knot sequences U and V. The polygonal surface will have
% (num - 1)^2 quadrilaterals.

% p=1;q=1;
% V=[0,0,0.5,1,1];
% U=[0,0,1,1];
% CP1 = [0 0 0;1 3 5];
% CP2 = [0 2 5;0 3 4];
% CP3 = [1 2 3;3 2 1];
% CP = cat(3,CP1,CP2,CP3);
% CP = shiftdim(CP,2);
% 
% num = 30;

% if (nargin ~= 6)
%     error('Incorrect number of arguments');
% end

n1 = size(CP,2);
n2 = size(CP,3);

X = zeros(num,num);
Y = zeros(num,num);
Z = zeros(num,num);

for j = 1 : num
  s = (j - 1)/(num - 1);
  for i = 1 : num
    t = (i - 1)/(num - 1);
    pt = pointonsurface(p,q,U,V,CP,t,s);
    X(i,j) = pt(1);
    Y(i,j) = pt(2);
    Z(i,j) = pt(3);
  end
end

% p = surf(X,Y,Z);

surf(X,Y,Z,'FaceColor',[0.2 0 0 ],'FaceAlpha',1, 'EdgeColor', 'w');
%shading interp

        % style 1 
%          p.EdgeColor=0.1*[1 1 1 ];
%          p.FaceAlpha=.75;
%          p.FaceColor='green';

% style 2
%         p.FaceColor = [0.5 0.5 0.75];
%         p.FaceAlpha= 1; %0.75 %.75;
%         p.EdgeColor = 'green'; %[0.5 0.5 0.5] % 'green'; %'interp';
%         p.LineStyle = '-' %'-'; %'none';'

hold on

% out = [X, Y, Z];

return