function  [kntvpc]=nrbisov(edv,nrbs,sdn)

k=nrbs.order(1);
l=nrbs.order(2);
U=nrbs.knots{1};
V=nrbs.knots{2};
m=nrbs.number(1)-1;
n=nrbs.number(2)-1;

cph=shiftdim(nrbs.coefs,1);

lv=findspan(n,l-1,edv,V)+1;

net_u=reshape(coxdeboor(permute(cph(:,:,:),[2 3 1]),...
    V,l,lv,edv),4,m+1)';

if length(find(U==0))==1 && length(find(U==1))==1 ...
    && length(find(V==0))==1 && length(find(V==1))==1
    % uniform B-Spline
    Bu=[1 4 1 0;-3 0 3 0;3 -6 3 0;-1 3 -3 1]/6;
    ptchnou=m-k+2; % patch #
    smpu=linspace(0,1,sdn+1);
    tmpub=[smpu.^0;smpu.^1;smpu.^2;smpu.^3]'*Bu; % [B_v]^T [v]^T
    kntvpc=zeros(ptchnou*sdn+1,3);
    for jj=l:(l+ptchnou-1)
        cpspan=net_u(jj-k+1:jj,:);
        smpidx=(jj-k)*sdn+(1:sdn+1);
        nvbc=tmpub*cpspan;
        kntvpc(smpidx,:)=nvbc(:,1:3)./repmat(nvbc(:,4),1,3);
    end
elseif length(find(U==0))==k && length(find(U==1))==k ...
        && length(find(V==0))==l && length(find(V==1))==l
    % clamped B-Spline
    unqU=unique(U);
    intlu=unqU(2:end)-unqU(1:end-1);
    
    if length(sdn)==1
        spnu=sdn*length(intlu);
    elseif length(sdn)==2
        spnu=sdn(2)*length(intlu);
    end
    intnu=round(spnu*intlu);
    sdnu=sdnpara(unqU,intnu);
    
    kntvpc=zeros(length(sdnu),3);
    count_u=0;
    
    for u=sdnu
        lu=findspan(m,k-1,u,U)+1;
        nubc=coxdeboor(net_u,U,k,lu,u);
        count_u=count_u+1;
        kntvpc(count_u,:)=nubc(:,1:3)./repmat(nubc(:,4),1,3);
    end
else
    error('Unrecognized B-Spline surface type')
end

%%% @@@@@@@@ Nested function @@@@@@@@
function sdnu=sdnpara(unqU,intnu)
sdnu=[];
for ii=1:length(intnu)
    insp_=linspace(unqU(ii),unqU(ii+1),intnu(ii)+1);
    if ii==1
        insp=insp_;
    else
        insp=insp_(2:end);
    end
    sdnu=[sdnu,insp];
end