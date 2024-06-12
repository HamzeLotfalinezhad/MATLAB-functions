function H=DFA(x,order)
X=cumsum(x-mean(x));
% X=transpose(X);
ii=0;
for i=2:100000
    u=2.^i;
    if u>length(x)
        break
    else
        ii=ii+1;
scale(ii)=u;
    end
end

mm=0;
for ns=1:length(scale)
segments(ns)=floor(length(X)/scale(ns));
% figure
% mm=mm+1;
% subplot(2,1,mm)
% plot(X,'-ob','markersize',5,'markerfacecolor','b');
for v=1:segments(ns)
Idx_start=((v-1)*scale(ns))+1;
Idx_stop=v*scale(ns);
Index{v,ns}=Idx_start:Idx_stop;
X_Idx=X(Index{v,ns});

C=polyfit(Index{v,ns},X(Index{v,ns}),order);
fit{v,ns}=polyval(C,Index{v,ns});
% internal plot
% hold on
% plot(Index{v,ns},fit{v,ns},'r','linewidth',2)
% xlabel('Samples');ylabel('amplitude');
% line([Idx_stop Idx_stop],[min(X) max(X)],'color',[0 0 0]);
% axis([0 length(X) min(X) max(X)])
% pause(0.1)
%
RMS{ns}(v)=sqrt(mean((X_Idx-fit{v,ns}).^2));
end
F(ns)=sqrt(mean(RMS{ns}.^2));
end
%%
C=polyfit(log2(scale),log2(F),1);
H=C(1);
if nargout ==0
RegLine=polyval(C,log2(scale));
plot(log2(scale),log2(F),'-ob','markersize',3,'markerfacecolor','b');
hold on
plot(log2(scale),RegLine,'r','linewidth',2)
xlabel('Log_{2}n');ylabel('Log_{2}F(n)');
grid on
end
end