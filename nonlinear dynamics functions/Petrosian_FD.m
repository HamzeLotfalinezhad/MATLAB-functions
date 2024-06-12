function FD_P=Petrosian_FD(x)
n=length(x);
x=NORM01(x);
dx=diff(x);

Nd=0;
for i=1:length(dx)-2
    if (dx(i)<dx(i+1) && dx(i+1)>dx(i+2)) || (dx(i)>dx(i+1) && dx(i+1)<dx(i+2))
        Nd=Nd+1;
    end
end

% FD_P=log10(n)/( log10(n)+log10( n/(n+0.4*Nd)) );
FD_P=(log10(n)/( log10(n)+log10( n/(n+0.4*Nd)) ));

end