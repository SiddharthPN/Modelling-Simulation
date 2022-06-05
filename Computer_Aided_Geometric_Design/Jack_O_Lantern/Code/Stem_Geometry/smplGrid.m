function SMG=smplGrid(limu,limv,dvn)
smbasu=linspace(limu(1),limu(2),dvn(1)+1)';
smbasv=linspace(limv(1),limv(2),dvn(2)+1)';
[smbgv,smbgu]=meshgrid(smbasv,smbasu);
SMG=cat(3,smbgu,smbgv);