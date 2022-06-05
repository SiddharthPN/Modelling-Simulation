function [vtx,tri]=readSTL(stlfile,isPlot)
%%% Read mesh information from ASCII STL file

%%% ====== Load vertices and facets normals =====
fp=fopen(stlfile,'r');
fgetl(fp);  %1st line: "solid SampSrfCld"
k=0;
while ~feof(fp)
    %     if rem(k,5000)==0
    %         k
    %     end
    tline=fgetl(fp); %1st line: "facet normal"
    N=sscanf(tline, '%*s %*s %f %f %f');
    if feof(fp)
        break;
    else
        fgetl(fp); % 2nd line "outer loop"
    end
    tline=fgetl(fp); % 1st vertex: "vertex (x) (y) (z)"
    A=sscanf(tline, '%*s %f %f %f');
    tline=fgetl(fp); % 2nd vertex: "vertex (x) (y) (z)"
    B=sscanf(tline, '%*s %f %f %f');
    tline=fgetl(fp); % 3rd vertex: "vertex (x) (y) (z)"
    C=sscanf(tline, '%*s %f %f %f');
    k=k+1;
    trifacet(k,:)=[A',B',C',N'];
    fgetl(fp); % "endloop"
    fgetl(fp); % "endfacet"
end
fclose(fp);

%%% ====== Establish connectivity graph =======
input=trifacet;
vtx(1,:)=input(1,1:3);
vtxidx=0;
tri=zeros(size(input,1),3);
vA=input(:,1:3);
vB=input(:,4:6);
vC=input(:,7:9);

for ii=1:size(input,1)
    if tri(ii,1)==0,
        tsp=vA(ii,:);
        vtxidx=vtxidx+1;
        [rowA]=find(vA(:,1)==tsp(1,1)&...
            vA(:,2)==tsp(1,2)&vA(:,3)==tsp(1,3));
        [rowB]=find(vB(:,1)==tsp(1,1)&...
            vB(:,2)==tsp(1,2)&vB(:,3)==tsp(1,3));
        [rowC]=find(vC(:,1)==tsp(1,1)&...
            vC(:,2)==tsp(1,2)&vC(:,3)==tsp(1,3));
        
        tri(rowA,1)=vtxidx;
        tri(rowB,2)=vtxidx;
        tri(rowC,3)=vtxidx;
        
        vtx(vtxidx,:)=tsp;
    end
    if tri(ii,2)==0,
        tsp=vB(ii,:);
        vtxidx=vtxidx+1;
        [rowA]=find(vA(:,1)==tsp(1,1)&...
            vA(:,2)==tsp(1,2)&vA(:,3)==tsp(1,3));
        [rowB]=find(vB(:,1)==tsp(1,1)&...
            vB(:,2)==tsp(1,2)&vB(:,3)==tsp(1,3));
        [rowC]=find(vC(:,1)==tsp(1,1)&...
            vC(:,2)==tsp(1,2)&vC(:,3)==tsp(1,3));
        
        tri(rowA,1)=vtxidx;
        tri(rowB,2)=vtxidx;
        tri(rowC,3)=vtxidx;
        
        vtx(vtxidx,:)=tsp;
    end
    if tri(ii,3)==0,
        tsp=vC(ii,:);
        vtxidx=vtxidx+1;
        [rowA]=find(vA(:,1)==tsp(1,1)&...
            vA(:,2)==tsp(1,2)&vA(:,3)==tsp(1,3));
        [rowB]=find(vB(:,1)==tsp(1,1)&...
            vB(:,2)==tsp(1,2)&vB(:,3)==tsp(1,3));
        [rowC]=find(vC(:,1)==tsp(1,1)&...
            vC(:,2)==tsp(1,2)&vC(:,3)==tsp(1,3));
        
        tri(rowA,1)=vtxidx;
        tri(rowB,2)=vtxidx;
        tri(rowC,3)=vtxidx;
        
        vtx(vtxidx,:)=tsp;
    end
end

%%% ======== Plot mesh if desired =========
if isPlot=='y'
    figure;set(gcf,'color','w');hold on;
    trisurf(tri,vtx(:,1),vtx(:,2),vtx(:,3),...
        'FaceColor',.8*[1,1,1]);
    axis equal tight off;view(3);
    camlight;lighting gouraud;
end