function genhurst=GHurstE(x,q,taomax)
% my generalized hurst exponent
ii=0;
for i=1:taomax
    y1=psr_deneme(x,2,i)';
    y2=mean(abs((y1(2,:)-y1(1,:))).^q);
    y3=abs(mean(x.^q));
    ii=ii+1;
    k(ii)=y2/y3;
    kx(ii)=i;
end
Logx=log(kx);
Logy=log(k);
p= polyfit(Logx,Logy,1);
genhurst=p(1);

    if nargout==0
    plot(Logx,Logy,'-ob','markersize',4,'markerfacecolor','b');
    f = polyval(p,Logx);
    hold on;plot(Logx,f,'r','linewidth',2)
    xlabel('ln \tau')
    ylabel(['ln K_',num2str(q),'(\tau) '])
    title('Generalized Hurst exponent')
    grid on
    end
end