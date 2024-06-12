function Mutual_Info=MI(X,Y)

L1=length(X);
L2=length(Y);

Px=hist(X,L1)/L1;
Py=hist(Y,L2)/L2;

% Px=hist(X,L1)/trapz(X);
% Py=hist(Y,L2)/trapz(Y);

[t1,t2] = ndgrid(1:L1, 1:L2);
[t11,t22] = ndgrid(X, Y);

Joint = cat(3,Px(t1).*Py(t2), t11, t22);


Joint_Pxy=Joint(:,:,1);
mPx=Joint(:,:,2);
mPy=Joint(:,:,3);



Mutual_Info=-nansum(nansum(Joint_Pxy.*log2(Joint_Pxy./(mPx.*mPy))));


end

