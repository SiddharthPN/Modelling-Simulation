function [kntupc]=nrbisou(edu,nrbs,varargin)

switch length(nrbs.order)
    case 1 %%% =============== NURBS Curve ===============
        k=nrbs.order(1);
        U=nrbs.knots;
        m=nrbs.number(1)-1;
        cph=shiftdim(nrbs.coefs,1);
        lu=findspan(m,k-1,edu,U)+1;
        kntph=coxdeboor(permute(cph(:,:,:),[1 3 2]),U,k,lu,edu);
        kntupc=kntph(1:3)/kntph(4);
        
    case 2 %%% =============== NURBS Surface ===============
        sdn=varargin{1};
        k=nrbs.order(1);
        l=nrbs.order(2);
        U=nrbs.knots{1};
        V=nrbs.knots{2};
        m=nrbs.number(1)-1;
        n=nrbs.number(2)-1;
        
        cph=shiftdim(nrbs.coefs,1);
        lu=findspan(m,k-1,edu,U)+1;
        net_v=reshape(coxdeboor(permute(cph(:,:,:),[1 3 2]),...
            U,k,lu,edu),4,n+1)';
            
        unftol=1e-6;
        ynunfu=std(unique(U(2:end)-U(1:end-1))); % u knots uniform?
        ynunfv=std(unique(V(2:end)-V(1:end-1))); % v knots uniform?
        if length(find(abs(U)<unftol))==1 ...
                && length(find(abs(U-1)<unftol))==1 ...
                && length(find(abs(V)<unftol))==1 ...
                && length(find(abs(V-1)<unftol))==1 ...
                && ynunfu<unftol && ynunfv<unftol
            % uniform B-Spline
            Bv=[1 4 1 0;-3 0 3 0;3 -6 3 0;-1 3 -3 1]/6;
            ptchnov=n-l+2; % patch #
            smpv=linspace(0,1,sdn+1);
            tmpvb=[smpv.^0;smpv.^1;smpv.^2;smpv.^3]'*Bv; % [B_v]^T [v]^T
            kntupc=zeros(ptchnov*sdn+1,3);
            for ii=k:(k+ptchnov-1)
                cpspan=net_v(ii-k+1:ii,:);
                smpidx=(ii-k)*sdn+(1:sdn+1);
                nvbc=tmpvb*cpspan;
                kntupc(smpidx,:)=nvbc(:,1:3)./repmat(nvbc(:,4),1,3);
            end
        elseif length(find(abs(U)<unftol))==k ...
                && length(find(abs(U-1)<unftol))==k ...
                && length(find(abs(V)<unftol))==l ...
                && length(find(abs(V-1)<unftol))==l ...
                && (ynunfu>unftol && ynunfv>unftol)
            % clamped B-Spline
            unqV=unique(V);
            intlv=unqV(2:end)-unqV(1:end-1);
            if length(sdn)==1
                spnv=sdn*length(intlv);
            elseif length(sdn)==2
                spnv=sdn(2)*length(intlv);
            end
            intnv=round(spnv*intlv);
            smplv=smplpara(unqV,intnv);
            
            kntupc=zeros(length(smplv),3);
            count_v=0;
            for v=smplv
                lv=findspan(n,l-1,v,V)+1;
                nvbc=coxdeboor(net_v,V,l,lv,v);
                count_v=count_v+1;
                kntupc(count_v,:)=nvbc(:,1:3)./repmat(nvbc(:,4),1,3);
            end
        else
            error('Unrecognized B-Spline surface type')
        end
        
        
    case 3 %%% =============== NURBS Volume ===============
        
    otherwise
        error('Unregconized NURBS geometry!');
end


%%% @@@@@@@@ Nested function @@@@@@@@
function sdnu=smplpara(unqU,intnu)
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