function plotNrbs(nrbs)
%%% Plot NURBS Surface

kntu=nrbs.knots{1};
kntv=nrbs.knots{2};
p=nrbs.order(1)-1;
q=nrbs.order(2)-1;
limu=[kntu(p+1),kntu(end-p)];
limv=[kntv(q+1),kntv(end-q)];  % XQ: chnage p+1 to q+1
dvn=[30,30];
SM=smplGrid(limu,limv,dvn);
NBS=smplNrbs(SM,nrbs);

%figure;
set(gcf,'color','w');hold on;
surf(NBS(:,:,1),NBS(:,:,2),NBS(:,:,3),...
    'edgecolor','w','facecolor',[0 0 0]);
% plotNrbsKnt(nrbs);
% plotNrbsCp(nrbs)
axis equal tight off;view(3);
camlight;lighting gouraud;
hold on; 

%%% #################################################
function plotNrbsCp(nrbs)
cph=shiftdim(nrbs.coefs,1);
cpc=cph(:,:,1:3)./repmat(cph(:,:,4),[1,1,3]);  % assuming CPs are homogeneous CP already. XQ
for ii=1:size(cpc,1) %%% CP in V direction
    plot3(cpc(ii,:,1),cpc(ii,:,2),cpc(ii,:,3),...
        'color','c','linestyle','-.','linewidth',1,...
        'marker','o','markersize',8,...
        'markeredgecolor','k', 'markerfacecolor','r');
end
for ii=1:size(cpc,2)  %%% CP in U direction
    plot3(cpc(:,ii,1),cpc(:,ii,2),cpc(:,ii,3),...
        'color','m','linestyle','--','linewidth',1,...
        'marker','o','markersize',8,...
        'markeredgecolor','k', 'markerfacecolor','r');
end
hold on;

function plotNrbsKnt(nrbs)
U=nrbs.knots{1};
V=nrbs.knots{2};
k=nrbs.order(1);
l=nrbs.order(2);
m=nrbs.number(1)-1; % cp#-1
n=nrbs.number(2)-1; % cp#-1
sdnu=30;sdnv=30;
ptchnou=m-k+2;
ptchnov=n-l+2;
kntallu=U((k+1):(k+ptchnou-1)); % knot point to plot
kntallv=V((l+1):(l+ptchnov-1)); % knot point to plot
if isempty(kntallu) % whole patch without knots
    kntpcu=[];
else
    for ii=1:length(kntallu)
        if ii==1
            tmps=nrbisou(kntallu(ii),nrbs,sdnu);
            kntpcu=zeros(size(tmps,1),3,length(kntallu));
        end
        kntpcu(:,:,ii)=nrbisou(kntallu(ii),nrbs,sdnu);
    end
end

if isempty(kntallv)
    kntpcv=[];
else
    for ii=1:length(kntallv)
        if ii==1
            tmps=nrbisov(kntallv(ii),nrbs,sdnv);
            kntpcv=zeros(size(tmps,1),3,length(kntallv));
        end
        kntpcv(:,:,ii)=nrbisov(kntallv(ii),nrbs,sdnv);
    end
end

%%%  Plot knot u-curves, v-curves
if ~isempty(kntpcu)
    for iii=1:size(kntpcu,3)
        pltpt=kntpcu(:,:,iii)';
        plot3(pltpt(1,:),pltpt(2,:),pltpt(3,:),...
            'linestyle','-','color','b','linewidth',1);
    end
end

if ~isempty(kntpcv)
    for iii=1:size(kntpcv,3)
        pltpt=kntpcv(:,:,iii)';
        plot3(pltpt(1,:),pltpt(2,:),pltpt(3,:),...
            'linestyle','-','color','b','linewidth',1);
    end
end
hold on;