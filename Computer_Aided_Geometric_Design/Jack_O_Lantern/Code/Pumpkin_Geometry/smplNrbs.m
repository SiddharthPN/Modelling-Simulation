function PUa=smplNrbs(UU,tnrbs)
%%% Sample points on NURBS

ns=length(tnrbs);
pdim=3; % output points dimension
%%% ------------- Parse/Check input -------------
if size(UU,4)~=ns
    error('Inconsisten shape number to sample!!!');
end
[npu,npv,~,~]=size(UU);
PUa=zeros(npu,npv,pdim,ns);
for ii=1:ns
    %%% ----------- Basic parameters ------------
    [p,q,U,V,kbu,kbv,ncpu,ncpv,cph]...
        =parseTSrf(tnrbs(ii));
    %%% ------- Check within valid domain -------
    smui=UU(:,:,1,ii);
    smvi=UU(:,:,2,ii);
    if any(any(smui<kbu(1))) || any(any(smui>kbu(2)))
        error('Out of proper U-knot span!!!');
    end
    if any(any(smvi<kbv(1))) || any(any(smvi>kbv(2)))
        error('Out of proper V-knot span!!!');
    end
    %%% ------- Evaluation w/ CoxDeBoor  --------
    for jju=1:npu
        for jjv=1:npv
            uij=smui(jju,jjv);
            vij=smvi(jju,jjv);
            evpc=evalCdb(uij,vij,p,q,U,V,ncpu,ncpv,cph);
            PUa(jju,jjv,:,ii)=evpc;
        end
    end
end

%%% #######################################################
function evpc=evalCdb(uij,vij,p,q,U,V,ncpu,ncpv,cph)
lu=findspan(ncpu-1,p,uij,U)+1;
lv=findspan(ncpv-1,q,vij,V)+1;
evptv=reshape(coxdeboor(permute(cph,...
    [1 3 2]),U,p+1,lu,uij),4,ncpv)';
evph=coxdeboor(evptv,V,q+1,lv,vij);
evpc=evph(1:3)/evph(4);

%%% ============== Parse NURBS Surface ===============
function [p,q,U,V,kbu,kbv,ncpu,ncpv,cph]=parseTSrf(nrbsi)
p=nrbsi.order(1)-1;
q=nrbsi.order(2)-1;
U=nrbsi.knots{1};
V=nrbsi.knots{2};
typu=idtfyKnts(p,U);
typv=idtfyKnts(q,V);
kbu=findPtchkb(typu,p,U);
kbv=findPtchkb(typv,q,V);
ncpu=nrbsi.number(1);
ncpv=nrbsi.number(2);
cph=shiftdim(nrbsi.coefs,1);

%%% =============== Identify knots type ===============
function typ=idtfyKnts(p,U)
%%% -------------------- INPUT ---------------------
%%% p: degree
%%% U: knot vector
%%% varargin: tol: distinguishing tolerance
%%% -------------------- OUTPUT ---------------------
%%% typ: knots type, uniform~'u' / clamped~'c'
kntdif=diff(U);
if any(kntdif<0)
    error('Knots are not non-decreasing') ;
end
tol=1e-6; % Tolerance to distinguish two neighbor knots
if std(kntdif)<=tol*(diff(U([1,end])))
    typ='u'; % Uniform knots
else % Non-uniform knots
    ynclamp1=all(kntdif(1:p)==0); % check head
    ynclamp2=all(kntdif(end-p+1:end)==0); % tail
    if ynclamp1 && ynclamp2
        typ='c'; % Clamped knots
    else
        error('Unrecognized knots type')
    end
end

%%% ============ Find patch knots bound =================
function kbd=findPtchkb(typ,p,U)
%%% Effective span index as patches
%%% Uniform and clamped are the same if multiplicity=0
if typ=='c' || typ=='C'
    kspnz=find(diff(U)~=0);
elseif typ=='u' || typ=='U'
    % all knot span # (may NOT=patch #)
    kspnz=(p+1):(length(U)-(p+1));
else
end
kbd=[U(kspnz(1)),U(kspnz(end)+1)];

