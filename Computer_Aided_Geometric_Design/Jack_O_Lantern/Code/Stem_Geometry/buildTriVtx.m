function [tri,vtx]=buildTriVtx(nrbs,dvn)
%%% dvn: sampling resolution
SM=smplGrid([0,1],[0,1],dvn);
PTS=smplNrbs(SM,nrbs);
[npu,npv,~]=size(PTS);
idxmtx=reshape(1:npu*npv,npu,npv);
vtx=reshape(permute(PTS,[3,1,2]),3,npu*npv)';
tri=zeros((npu-1)*(npv-1),3);
wcount=0;
for ki=1:npu-1
    for kj=1:npv-1
        wcount=wcount+1;
        idquad=idxmtx(ki:ki+1,kj:kj+1);
        wldx=(2*wcount-1):(2*wcount);
        tri(wldx,:)=[idquad([1,2,3]);idquad([2,4,3])];
    end
end