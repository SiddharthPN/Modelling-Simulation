%%% =================== find knot span =================
function s = findspan(n,p,u,U)
if (u==U(n+2)), s=n; return,  end
low = p;
high = n + 1;
mid = floor((low + high) / 2);
while (u < U(mid+1) || u >= U(mid+2))
    if (u < U(mid+1))
        high = mid;
    else
        low = mid;
    end
    mid = floor((low + high) / 2);
end
s = mid;